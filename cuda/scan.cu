// JUNO (Just Use Noise Once), a GPU seed scanner for Minecraft 26.3.
// Seeds go through the gate first (on the GPU, or in hostgate.c with --cpu-gate), then the temperature, humidity and probe kernels.
// The cascade kernel counts the biomes for whatever is left, and the CPU checks the hits again at the end.
#define JUNO_VERSION "1.2"
#define BIOME_COUNT 52                   // surface biomes we score
#define TEMPERATURE_THREADS 128          // Threads in a block for the temperature kernel
#define STREAM_COUNT 5                   // Batches the GPU works on at once, each one gets a CUDA stream. Tried a few on my machine and 5 worked best, --streams changes it
#define BATCH_SIZE (1L << 22)            // The max number of gated indexes in a batch
#define RECORD_CAPACITY (BATCH_SIZE / 4) // How many temperature records a batch can hold
#define MIN_CELL_EVENNESS 0.9425f        // The evenness of the temperature and humidity cells a seed needs
#define LOWEST_MIN_SENTS 0.905           // GPU filters are tuned for hits with a SENTS score of at least this
#define LOWEST_MIN_ARBITRATIONS (100.0 * LOWEST_MIN_SENTS * LOWEST_MIN_SENTS) // same thing but for ARBITRATIONS
#define RESULTS_FOLDER "../results"
#define CHECKPOINT_PATH RESULTS_FOLDER "/gpu_progress.txt"
#define SPEED_PATH RESULTS_FOLDER "/gpu_speed.txt" // make status reads the speed from here
#define LOCK_PATH "/tmp/juno-scan.lock" // One scanner at a time on a computer
#define HITS_PATH RESULTS_FOLDER "/gpu_hits.jsonl"
#define CUSTOM_SEED_PATH RESULTS_FOLDER "/custom_seed.txt"
#define CHECKPOINT_TAG "juno-gpu-v1"
#include "biome.cuh"
#include "lut.cuh"
#include "bbox.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <future> // std::async
#include <chrono>
#include <ctime>
#include <cstring>
#include <utility>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <sys/stat.h> // mkdir
#include <sys/file.h> // flock
#include <fcntl.h>
#include <unistd.h>
#include <csignal>
#include <cerrno>
#include <climits> // LONG_MAX and ULLONG_MAX
#include <cctype>
#include <set>
#include <string>
#include <random>
#include <atomic>
#include <deque>
#include "hostgate.h"
extern "C" {
#include "biomenoise.h"
}

__constant__ int8_t BIOME_SLOTS[64]; // Slot in the biome amounts for each lookup table id (-1 if we don't score that biome)

#define SAMPLE_COUNT 81 // The temperature, humidity and probe kernels sample a 9 by 9 grid

// The log weights for the temperature bands and cells, uploadHistogramTables works them out
__constant__ double BAND_LOG_WEIGHTS[5];
__constant__ double CELL_LOG_WEIGHTS[25];
__constant__ double INVERSE_LOG_TOTAL;
const double TEMP_EDGES[4] = {-0.45, -0.15, 0.2, 0.55}; // temperature band edges
const double HUMID_EDGES[4] = {-0.35, -0.1, 0.1, 0.3};  // humidity band edges
#define CASCADE_THREADS 512 // cascade threads per seed (tuned by hand, like the other block sizes)
#define SHIFT_OCTAVES 6     // The shift octaves, they come first in the octave array
#define LEVEL_COUNT 9       // One more than the last cascade level, levels go from 1 to 8
#define LAST_RANK_LEVEL 5   // The last level that uses the rank score
#define LAST_MAP_LEVEL 5    // Last level that puts all of it's cells on the coarse map
// Distance between the cells on each level. Nothing reads STRIDE[0] now, the cascade used to start with 16 cells but starting with 64 was faster
__constant__ int STRIDE[LEVEL_COUNT] = {256, 128, 64, 32, 16, 8, 4, 2, 1};

// Proximity gate, the biomes and their boxes are in bbox.h
#define PROX_DISTANCE_SQ 4000000LL // how close a biome box has to be, squared
#define PROX_LEVEL 3               // The cascade level where we check the gate

// splitmix64 turns a stream index into a seed. hostgate.c does the same thing
DEV uint64_t streamSeed(uint64_t index) {
    uint64_t mixed = index * 0x9E3779B97F4A7C15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    return mixed ^ (mixed >> 31);
}

#include "cell_score.inc"
#include "temperature_score.inc"
#include "probe_score.inc"

#include "selector_cache.cuh"
#include "gpu_gate.cuh"

// The temperature kernel throws out seeds before we build humidity and saves a record for the humidity kernel. Cell evenness can't be
// higher than the temperature evenness, so MIN_CELL_EVENNESS works on temperature too
__constant__ double MIN_TEMPERATURE_EVENNESS = 0.968; // temperature evenness a seed needs (if MIN_CELL_EVENNESS is lower)
#define RECORD_WORDS 11                // Size of a temperature record (in 64 bit words)
#define PERSISTENCE (32.0 / 63)        // PERSISTENCE_START[6], since temperature and humidity have 6 amplitudes

// The histograms get packed into 64 bit numbers, one byte per bin
DEV int packedAmount(uint64_t packed, int bin) {
    return (int) ((packed >> (8 * bin)) & 255);
}

DEV int binAmount(uint64_t low, uint64_t high, int bin) {
    if (bin < 8) {
        return packedAmount(low, bin);
    }
    return packedAmount(high, bin & 7);
}

// Returns how many of the GPU gate's indexes are left from chunkFirst on (below zero past the end of the list)
DEV long gatedLeft(const uint32_t *gatedCount, uint32_t chunkFirst) {
    return (long) min(*gatedCount, (uint32_t) BATCH_SIZE) - (long) chunkFirst;
}

/**
 * @brief This kernel shuffles the two temperature octaves in the shared tables and saves their corner slices for the temperature
 * kernel
 * 
 * @param indexes The stream indexes
 * @param indexCount How many indexes there are
 * @param built Where the corner slices go
 * @param gatedCount The number of indexes the GPU gate let through, NULL with --cpu-gate
 * @param chunkFirst Where this chunk starts in the list
 */
__global__ void __launch_bounds__(TABLE_THREADS, TABLE_BLOCKS)
temperatureBuildKernel(const uint64_t *indexes, int indexCount, uint64_t *built, const uint32_t *gatedCount = NULL, uint32_t chunkFirst = 0) {
    __shared__ uint32_t tableWords[64 * TABLE_THREADS];
    if (gatedCount) {
        long left = gatedLeft(gatedCount, chunkFirst);
        indexCount = (int) (left < indexCount ? left : indexCount);
    }
    // All the threads help with the shared tables, even the ones without a seed
    int blockFirst = blockIdx.x * TABLE_THREADS;
    if (blockFirst >= indexCount) {
        return;
    }
    int i = blockFirst + threadIdx.x;
    bool hasSeed = i < indexCount;

    uint64_t halfLow[2], halfHigh[2];
    climateHalves(streamSeed(indexes[hasSeed ? i : blockFirst]), 0x5c7e6b29735f0d7fULL, 0xf7d86f1bbc734988ULL, halfLow, halfHigh); // temperature salt
    uint8_t *table = tableStart(tableWords);
#pragma unroll 1
    for (int half = 0; half < 2; half++) {
        uint64_t slices[4];
        HalfHeader header;
        buildHalf(tableWords, table, half ? halfLow[1] : halfLow[0], half ? halfHigh[1] : halfHigh[0], 2, half, 10, 3, slices, &header);
        if (hasSeed) {
            saveBuiltHalf(built + i, 3, half, slices, header);
        }
    }
}

// Float copies of the score weights and log weights, uploadFloatTables fills them from the double ones
__constant__ float FLOAT_TEMPERATURE_SCORE_WEIGHTS[49];
__constant__ float FLOAT_TEMPERATURE_SCORE_BIAS;
__constant__ float FLOAT_TEMPERATURE_SCORE_THRESHOLD;
__constant__ float FLOAT_BAND_LOG_WEIGHTS[5];
__constant__ float FLOAT_CELL_LOG_WEIGHTS[25];
__constant__ float FLOAT_INVERSE_LOG_TOTAL;
__constant__ float FLOAT_MIN_TEMPERATURE_EVENNESS;

// How much we loosen the float cuts to make up for the rounding
#define FLOAT_SLACK 2e-5f

// What a temperature sample adds to the histograms, looked up by where it lands between -1 and 1 in steps of 1/400.
// The band edges, bin edges and near-edge limits all line up with these steps. uploadTemperatureTable fills it in
#define HISTOGRAM_CELLS 800
__device__ uint64_t HISTOGRAM_TABLE[HISTOGRAM_CELLS * 3];

// Like packedAmount and binAmount but with 7 bit amounts
DEV int tableAmount(uint64_t packed, int field) {
    return (int) ((packed >> (7 * field)) & 127);
}

DEV int tableBinAmount(uint64_t low, uint64_t high, int bin) {
    if (bin < 8) {
        return tableAmount(low, bin);
    }
    return tableAmount(high, bin & 7);
}

// The share of a histogram bin, the square root of the share, and share * log(share)
DEV float shareOf(int count) {
    return (float) count * (1.0f / SAMPLE_COUNT);
}

DEV float rootShareOf(int count) {
    float share = shareOf(count);
    return count ? share * rsqrtf(share) : 0.0f;
}

DEV float shareLogOf(int count) {
    float share = shareOf(count);
    return count ? share * __logf(share) : 0.0f;
}

/**
 * @brief Checks the temperature of a seed, and saves a record for the humidity kernel if the seed passes
 * 
 * @param seed
 * @param built What the temperature build kernel saved for this seed
 * @param capacity How many records fit in the output
 * @param outputCount Counts the seeds that passed
 * @param records Where the record gets saved
 * @param histogramTable HISTOGRAM_TABLE, copied to shared memory
 * @param gradients The gradient table in shared memory
 */
DEV void filterTemperature(uint64_t seed, const uint64_t *built, uint32_t capacity, uint32_t *outputCount, uint64_t *records, const uint64_t *histogramTable, const GradientTable *gradients) {
    // The temperature bands of the samples, 3 bits a sample and two rows to a word
    uint64_t rowBands[5] = {0, 0, 0, 0, 0};

    // This just uses the lowest octave of the Perlin noises
    uint64_t slices[2][4];
    HalfHeader halves[2];
    GridSteps columns[2];
    uint32_t startRow[2];
#pragma unroll
    for (int half = 0; half < 2; half++) {
        readBuiltHalf(built, 3, half, slices[half], &halves[half]);
        fillGridSteps(&columns[half], half, 10, halves[half].offsetX);
        startRow[half] = (halves[half].offsetZ + (uint32_t) GRID_POSITIONS[half][10][0]) >> 24;
    }

    uint64_t bandAmounts = 0; // 7 bit amounts, the bands and then the near-edge amounts
    uint64_t tempBinsLow = 0, tempBinsHigh = 0;
    float tempSum = 0, tempSquareSum = 0;

    for (int row = 0; row < 9; row++) {
        // The cells this row of samples can land in, the first half reaches two and the second three
        RowCell rowCells[2][3];
#pragma unroll
        for (int half = 0; half < 2; half++) {
            // Where this row is in z, and the corner slices on either side of it
            uint32_t noise = halves[half].offsetZ + (uint32_t) GRID_POSITIONS[half][10][row];
            int rowCell = (int) (uint8_t) ((noise >> 24) - startRow[half]);
            float rowFraction = fractionOf(noise);
            float rowFade = fade(rowFraction);
            uint32_t nearZ = (uint32_t) slices[half][0];
            uint32_t farZ = (uint32_t) slices[half][1];
            if (rowCell == 1) {
                nearZ = (uint32_t) slices[half][1];
                farZ = (uint32_t) slices[half][2];
            } else if (rowCell == 2) {
                nearZ = (uint32_t) slices[half][2];
                farZ = (uint32_t) slices[half][3];
            }
#pragma unroll
            for (int cell = 0; cell < 3; cell++) {
                if (half == 0 && cell == 2) {
                    continue;
                }
                collapseRowCell(nearZ, farZ, cell, halves[half].yFraction, halves[half].yFade, rowFraction, rowFade, &rowCells[half][cell], gradients);
            }
        }

        uint32_t bandsThisRow = 0;
#pragma unroll
        for (int column = 0; column < 9; column++) {
            float halfValues[2];
#pragma unroll
            for (int half = 0; half < 2; half++) {
                // The first half never lands in the third cell
                int cell = columns[half].cell[column];
                RowCell picked = rowCells[half][0];
                if (cell == 1) {
                    picked = rowCells[half][1];
                } else if (half == 1 && cell == 2) {
                    picked = rowCells[half][2];
                }
                float fractionX = columns[half].fraction[column];
                float nearX = fmaf(picked.nearSlope, fractionX, picked.nearConstant);
                float farX = fmaf(picked.farSlope, fractionX - 1.0f, picked.farConstant);
                halfValues[half] = interpolate(columns[half].fade[column], nearX, farX);
            }
            // The two halves have the same amplitude
            float temperature = (halfValues[0] + halfValues[1]) * (float) (1.5 * PERSISTENCE * (15.0 / 12));

            int cell = min(max((int) floorf(fmaf(temperature, 400.0f, 400.0f)), 0), HISTOGRAM_CELLS - 1);
            const uint64_t *entry = histogramTable + 3 * cell;
            uint64_t binsLow = entry[1];
            bandAmounts += entry[0];
            tempBinsLow += binsLow;
            tempBinsHigh += entry[2];
            bandsThisRow |= (uint32_t) (binsLow >> 56) << (3 * column);

            tempSum += temperature;
            tempSquareSum += temperature * temperature;
        }
        uint64_t shifted = (uint64_t) bandsThisRow << (27 * (row & 1));
#pragma unroll
        for (int word = 0; word < 5; word++) {
            rowBands[word] |= (row >> 1) == word ? shifted : 0;
        }
    }

    float entropy = 0;
#pragma unroll
    for (int band = 0; band < 5; band++) {
        int count = tableAmount(bandAmounts, band);
        entropy += shareOf(count) * FLOAT_BAND_LOG_WEIGHTS[band] - shareLogOf(count);
    }
    float tempEvenness = entropy * FLOAT_INVERSE_LOG_TOTAL;
    if (tempEvenness < FLOAT_MIN_TEMPERATURE_EVENNESS) {
        return;
    }

    // The temperature score guesses whether or not the cell score would pass the seed, saves us building humidity for nothing
    float tempMean = tempSum * (1.0f / SAMPLE_COUNT);
    float tempVariance = tempSquareSum * (1.0f / SAMPLE_COUNT) - tempMean * tempMean;
    float score = FLOAT_TEMPERATURE_SCORE_BIAS + FLOAT_TEMPERATURE_SCORE_WEIGHTS[10] * tempEvenness + FLOAT_TEMPERATURE_SCORE_WEIGHTS[11] * tempMean
                  + FLOAT_TEMPERATURE_SCORE_WEIGHTS[12] * sqrtf(fmaxf(tempVariance, 0.0f));
#pragma unroll
    for (int i = 0; i < 5; i++) {
        int count = tableAmount(bandAmounts, i);
        score += FLOAT_TEMPERATURE_SCORE_WEIGHTS[i] * shareOf(count) + FLOAT_TEMPERATURE_SCORE_WEIGHTS[5 + i] * rootShareOf(count);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        score += FLOAT_TEMPERATURE_SCORE_WEIGHTS[13 + i] * shareOf(tableAmount(bandAmounts, 5 + i));
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int count = tableBinAmount(tempBinsLow, tempBinsHigh, i);
        score += FLOAT_TEMPERATURE_SCORE_WEIGHTS[17 + i] * shareOf(count) + FLOAT_TEMPERATURE_SCORE_WEIGHTS[33 + i] * rootShareOf(count);
    }

    if (score < FLOAT_TEMPERATURE_SCORE_THRESHOLD) {
        return;
    }

    // Hand the seed over to the humidity kernel
    uint32_t slot = atomicAdd(outputCount, 1u);
    if (slot < capacity) {
        uint64_t *output = records + (uint64_t) slot * RECORD_WORDS;
        output[0] = seed;
        output[1] = rowBands[0];
        output[2] = rowBands[1];
        output[3] = rowBands[2];
        output[4] = rowBands[3];
        output[5] = rowBands[4];
        output[6] = bandAmounts; // the near-edge amounts are in here too
        output[7] = 0;           // nothing uses this word now
        output[8] = tempBinsLow;
        output[9] = tempBinsHigh;
        output[10] = __float_as_uint(tempSum) | (uint64_t) __float_as_uint(tempSquareSum) << 32;
    }
}

DEV int cellAmount(uint64_t cellAmounts0, uint64_t cellAmounts1, uint64_t cellAmounts2, uint64_t cellAmounts3, int cell) {
    if (cell < 8) {
        return packedAmount(cellAmounts0, cell);
    } else if (cell < 16) {
        return packedAmount(cellAmounts1, cell & 7);
    } else if (cell < 24) {
        return packedAmount(cellAmounts2, cell & 7);
    }
    return packedAmount(cellAmounts3, cell & 7);
}

// Unpacks the record filterTemperature saved (the layout is at the end of that function)
DEV void unpackRecord(const uint64_t *record, uint64_t rowBands[5], int tempNearEdge[4], uint64_t *tempBinsLow,
                      uint64_t *tempBinsHigh, float *tempSum, float *tempSquareSum) {
    for (int i = 0; i < 5; i++) {
        rowBands[i] = record[1 + i];
    }
    for (int edge = 0; edge < 4; edge++) {
        tempNearEdge[edge] = tableAmount(record[6], 5 + edge);
    }
    *tempBinsLow = record[8];
    *tempBinsHigh = record[9];
    *tempSum = __uint_as_float((uint32_t) record[10]);
    *tempSquareSum = __uint_as_float((uint32_t) (record[10] >> 32));
}

/**
 * @brief This kernel shuffles the two humidity octaves in the shared tables for the seeds in the temperature records, and saves
 * their corner slices
 * 
 * @param records The records from the temperature kernel
 * @param recordCount The number of records
 * @param built Where the corner slices go
 */
__global__ void __launch_bounds__(TABLE_THREADS, TABLE_BLOCKS)
humidityBuildKernel(const uint64_t *records, int recordCount, uint64_t *built) {
    __shared__ uint32_t tableWords[64 * TABLE_THREADS];
    int blockFirst = blockIdx.x * TABLE_THREADS;
    if (blockFirst >= recordCount) {
        return;
    }
    int i = blockFirst + threadIdx.x;
    bool hasSeed = i < recordCount;

    uint64_t halfLow[2], halfHigh[2];
    climateHalves(records[(size_t) (hasSeed ? i : blockFirst) * RECORD_WORDS], 0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL, halfLow, halfHigh); //humidity salt
    uint8_t *table = tableStart(tableWords);
#pragma unroll 1
    for (int half = 0; half < 2; half++) {
        uint64_t slices[7];
        HalfHeader header;
        buildHalf(tableWords, table, half ? halfLow[1] : halfLow[0], half ? halfHigh[1] : halfHigh[0], 4, half, 8, 6, slices, &header);
        if (hasSeed) {
            saveBuiltHalf(built + i, 6, half, slices, header);
        }
    }
}

__constant__ float FLOAT_CELL_SCORE_WEIGHTS[128];
__constant__ float FLOAT_CELL_SCORE_BIAS;
__constant__ float FLOAT_CELL_SCORE_THRESHOLD;
__constant__ float FLOAT_PROBE_SCORE_WEIGHTS[200];
__constant__ float FLOAT_PROBE_SCORE_BIAS;
__constant__ float FLOAT_MIN_CELL_EVENNESS;
__constant__ float FLOAT_PROBE_SCORE_THRESHOLD;

// The humidity band edges as floats, and the lowest and highest float within 0.03 of an edge. uploadHistogramTables works
// them out
__constant__ float HUMID_BAND_EDGES[4];
__constant__ float HUMID_NEAR_LOW[4];
__constant__ float HUMID_NEAR_HIGH[4];

/**
 * @brief Samples humidity for the seed in a record, then checks the cell evenness and the cell score
 * 
 * @param record The record from the temperature kernel
 * @param built What the humidity build kernel saved for this seed
 * @param capacity Size of the output buffers
 * @param outputCount The number of seeds that passed
 * @param outputSeeds Where the seeds that pass get written
 * @param outputProbe The start of the probe score for those seeds
 */
DEV void filterHumidity(const uint64_t *record, const uint64_t *built, uint32_t capacity, uint32_t *outputCount, uint64_t *outputSeeds, double *outputProbe) {
    uint64_t seed = record[0];
    uint64_t rowBands[5];
    uint64_t tempBinsLow, tempBinsHigh;
    int tempNearEdge[4];
    float tempSum, tempSquareSum;
    unpackRecord(record, rowBands, tempNearEdge, &tempBinsLow, &tempBinsHigh, &tempSum, &tempSquareSum);

    // Only the lowest octave again, but humidity needs a bigger block of lattice cells than temperature
    uint64_t slices[2][7];
    HalfHeader halves[2];
    GridSteps columns[2];
    uint32_t startRow[2];
#pragma unroll
    for (int half = 0; half < 2; half++) {
        readBuiltHalf(built, 6, half, slices[half], &halves[half]);
        fillGridSteps(&columns[half], half, 8, halves[half].offsetX);
        startRow[half] = (halves[half].offsetZ + (uint32_t) GRID_POSITIONS[half][8][0]) >> 24;
    }

    // 25 temperature/humidity cells, so that histogram takes four of these
    uint64_t cellAmounts0 = 0, cellAmounts1 = 0, cellAmounts2 = 0, cellAmounts3 = 0;
    uint64_t humidBinsLow = 0, humidBinsHigh = 0;
    int humidNearEdge[4] = {0, 0, 0, 0};
    float humidSum = 0, humidSquareSum = 0;

    for (int row = 0; row < 9; row++) {
        uint64_t bandWord = rowBands[0];
#pragma unroll
        for (int word = 1; word < 5; word++) {
            bandWord = (row >> 1) == word ? rowBands[word] : bandWord;
        }
        uint32_t bandsThisRow = (uint32_t) (bandWord >> (27 * (row & 1)));

        // Where this row is in z, and the corner slices on either side of it
        float rowFraction[2], rowFade[2];
        uint64_t nearZ[2], farZ[2];
#pragma unroll
        for (int half = 0; half < 2; half++) {
            uint32_t noise = halves[half].offsetZ + (uint32_t) GRID_POSITIONS[half][8][row];
            int rowCell = (int) (uint8_t) ((noise >> 24) - startRow[half]);
            rowFraction[half] = fractionOf(noise);
            rowFade[half] = fade(rowFraction[half]);
            nearZ[half] = slices[half][0];
            farZ[half] = slices[half][1];
#pragma unroll
            for (int z = 1; z < 6; z++) {
                nearZ[half] = rowCell == z ? slices[half][z] : nearZ[half];
                farZ[half] = rowCell == z ? slices[half][z + 1] : farZ[half];
            }
        }

#pragma unroll
        for (int column = 0; column < 9; column++) {
            float halfValues[2];
#pragma unroll
            for (int half = 0; half < 2; half++) {
                int shift = 8 * columns[half].cell[column];
                halfValues[half] = sampleCorners(nearZ[half] >> shift, farZ[half] >> shift, columns[half].fraction[column], halves[half].yFraction,
                                                 rowFraction[half], columns[half].fade[column], halves[half].yFade, rowFade[half]);
            }
            float humidity = (halfValues[0] + halfValues[1]) * (float) (1.0 * PERSISTENCE * (10.0 / 9));
            int humidBand = (humidity >= HUMID_BAND_EDGES[0]) + (humidity >= HUMID_BAND_EDGES[1]) + (humidity >= HUMID_BAND_EDGES[2])
                          + (humidity >= HUMID_BAND_EDGES[3]);

            int cell = (int) ((bandsThisRow >> (3 * column)) & 7) * 5 + humidBand;
            int word = cell >> 3;
            uint64_t cellBit = 1ULL << (8 * (cell & 7));
            cellAmounts0 += word == 0 ? cellBit : 0;
            cellAmounts1 += word == 1 ? cellBit : 0;
            // iiiiii've gooooot question marks all around me all around
            cellAmounts2 += word == 2 ? cellBit : 0;
            cellAmounts3 += word == 3 ? cellBit : 0;

            humidSum += humidity;
            humidSquareSum += humidity * humidity;

            int bin = (int) floorf(8.0f * humidity) + 8;
            bin = min(max(bin, 0), 15);
            uint64_t binBit = 1ULL << (8 * (bin & 7));
            humidBinsLow += bin < 8 ? binBit : 0;
            humidBinsHigh += bin < 8 ? 0 : binBit;

#pragma unroll
            for (int edge = 0; edge < 4; edge++) {
                humidNearEdge[edge] += humidity >= HUMID_NEAR_LOW[edge] && humidity <= HUMID_NEAR_HIGH[edge];
            }
        }
    }

    float entropy = 0;
#pragma unroll
    for (int cell = 0; cell < 25; cell++) {
        int count = cellAmount(cellAmounts0, cellAmounts1, cellAmounts2, cellAmounts3, cell);
        entropy += shareOf(count) * FLOAT_CELL_LOG_WEIGHTS[cell] - shareLogOf(count);
    }

    float cellEvenness = entropy * FLOAT_INVERSE_LOG_TOTAL;
    if (cellEvenness < FLOAT_MIN_CELL_EVENNESS) {
        return;
    }

    float tempMean = tempSum * (1.0f / SAMPLE_COUNT);
    float humidMean = humidSum * (1.0f / SAMPLE_COUNT);
    float tempDeviation = sqrtf(fmaxf(tempSquareSum * (1.0f / SAMPLE_COUNT) - tempMean * tempMean, 0.0f));
    float humidDeviation = sqrtf(fmaxf(humidSquareSum * (1.0f / SAMPLE_COUNT) - humidMean * humidMean, 0.0f));

    // Add up the cell score. The probe score starts from the same numbers so it gets added up here too
    float score = FLOAT_CELL_SCORE_BIAS + FLOAT_CELL_SCORE_WEIGHTS[50] * cellEvenness;
    float probeScore = FLOAT_PROBE_SCORE_BIAS + FLOAT_PROBE_SCORE_WEIGHTS[50] * cellEvenness;
#pragma unroll
    for (int cell = 0; cell < 25; cell++) {
        int count = cellAmount(cellAmounts0, cellAmounts1, cellAmounts2, cellAmounts3, cell);
        float share = shareOf(count);
        float rootShare = rootShareOf(count);
        score += FLOAT_CELL_SCORE_WEIGHTS[cell] * share + FLOAT_CELL_SCORE_WEIGHTS[25 + cell] * rootShare;
        probeScore += FLOAT_PROBE_SCORE_WEIGHTS[cell] * share + FLOAT_PROBE_SCORE_WEIGHTS[25 + cell] * rootShare;
    }

    score += FLOAT_CELL_SCORE_WEIGHTS[51] * tempMean + FLOAT_CELL_SCORE_WEIGHTS[52] * tempDeviation + FLOAT_CELL_SCORE_WEIGHTS[53] * humidMean + FLOAT_CELL_SCORE_WEIGHTS[54] * humidDeviation;
    probeScore += FLOAT_PROBE_SCORE_WEIGHTS[51] * tempMean + FLOAT_PROBE_SCORE_WEIGHTS[52] * tempDeviation + FLOAT_PROBE_SCORE_WEIGHTS[53] * humidMean + FLOAT_PROBE_SCORE_WEIGHTS[54] * humidDeviation;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float tempEdge = shareOf(tempNearEdge[i]);
        float humidEdge = shareOf(humidNearEdge[i]);
        score += FLOAT_CELL_SCORE_WEIGHTS[56 + i] * tempEdge + FLOAT_CELL_SCORE_WEIGHTS[60 + i] * humidEdge;
        probeScore += FLOAT_PROBE_SCORE_WEIGHTS[56 + i] * tempEdge + FLOAT_PROBE_SCORE_WEIGHTS[60 + i] * humidEdge;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int tempCount = tableBinAmount(tempBinsLow, tempBinsHigh, i);
        int humidCount = binAmount(humidBinsLow, humidBinsHigh, i);
        float tempShare = shareOf(tempCount);
        float humidShare = shareOf(humidCount);
        float tempRoot = rootShareOf(tempCount);
        float humidRoot = rootShareOf(humidCount);
        score += FLOAT_CELL_SCORE_WEIGHTS[64 + i] * tempShare + FLOAT_CELL_SCORE_WEIGHTS[80 + i] * tempRoot + FLOAT_CELL_SCORE_WEIGHTS[96 + i] * humidShare + FLOAT_CELL_SCORE_WEIGHTS[112 + i] * humidRoot;
        probeScore += FLOAT_PROBE_SCORE_WEIGHTS[64 + i] * tempShare + FLOAT_PROBE_SCORE_WEIGHTS[80 + i] * tempRoot + FLOAT_PROBE_SCORE_WEIGHTS[96 + i] * humidShare + FLOAT_PROBE_SCORE_WEIGHTS[112 + i] * humidRoot;
    }

    if (score < FLOAT_CELL_SCORE_THRESHOLD) {
        return;
    }

    uint32_t slot = atomicAdd(outputCount, 1u);
    if (slot < capacity) {
        outputSeeds[slot] = seed;
        outputProbe[slot] = probeScore;
    }
}

#define TEMPERATURE_BLOCKS 4 // Blocks of the temperature kernel on an SM at once. Tried a few and 4 was the fastest

/**
 * @brief This kernel checks the temperature for a batch of stream indexes
 * 
 * @param indexes The stream indexes
 * @param indexCount How many indexes there are
 * @param built What the temperature build kernel saved for these indexes
 * @param capacity Room in the records buffer
 * @param outputCount Counts the seeds that passed (this can go past capacity)
 * @param records Output records for the seeds that pass
 * @param gatedCount The number of indexes the GPU gate let through, NULL with --cpu-gate
 * @param chunkFirst Where this chunk starts in the list
 */
__global__ void __launch_bounds__(TEMPERATURE_THREADS, TEMPERATURE_BLOCKS)
temperatureKernel(const uint64_t *indexes, int indexCount, const uint64_t *built,
                  uint32_t capacity, uint32_t *outputCount, uint64_t *records, const uint32_t *gatedCount = NULL, uint32_t chunkFirst = 0) {
    // Leave right away if this chunk is past the end of the gate's list
    if (gatedCount && (long) blockIdx.x * blockDim.x >= gatedLeft(gatedCount, chunkFirst)) {
        return;
    }
    __shared__ uint64_t histogramTable[HISTOGRAM_CELLS * 3];
    __shared__ GradientTable gradients;
    for (int word = threadIdx.x; word < HISTOGRAM_CELLS * 3; word += blockDim.x) {
        histogramTable[word] = HISTOGRAM_TABLE[word];
    }
    fillGradientTable(&gradients);
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (gatedCount) {
        long left = gatedLeft(gatedCount, chunkFirst);
        indexCount = (int) (left < indexCount ? left : indexCount);
    }
    if (i >= indexCount) {
        return;
    }
    filterTemperature(streamSeed(indexes[i]), built + i, capacity, outputCount, records, histogramTable, &gradients);
}

#define HUMIDITY_THREADS 64

/**
 * @brief This kernel checks the humidity of the seeds that passed the temperature kernel
 * 
 * @param records The records from the temperature kernel
 * @param recordCount The number of records
 * @param built What the humidity build kernel saved for these records
 * @param capacity Size of the output buffers
 * @param outputCount Seeds that passed
 * @param outputSeeds Output for the seeds that pass
 * @param outputProbe Where the start of their probe score goes
 */
__global__ void __launch_bounds__(HUMIDITY_THREADS)
humidityKernel(const uint64_t *records, int recordCount, const uint64_t *built,
               uint32_t capacity, uint32_t *outputCount, uint64_t *outputSeeds, double *outputProbe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= recordCount) {
        return;
    }
    filterHumidity(records + (uint64_t) i * RECORD_WORDS, built + i, capacity, outputCount, outputSeeds, outputProbe);
}

// The probe kernel adds continentalness and erosion onto the probe score from the humidity kernel,
// using the lowest octaves at the same sample points as the other kernels
#define LAST_PROBE_CLIMATE 4 // Erosion

// The corner slices, offsets and y fraction of a probe octave
struct ProbeOctave {
    uint64_t slices[7];
    HalfHeader header;
    uint32_t startColumn;
    uint32_t startRow;
};

// Shuffles one probe octave and gets its corner slices
DEV void buildProbeOctave(uint32_t *tableWords, uint8_t *table, uint64_t halfLow, uint64_t halfHigh, int saltIndex, int half, int lacunarityExponent, int side,
                          ProbeOctave *octave) {
    buildHalf(tableWords, table, halfLow, halfHigh, saltIndex, half, lacunarityExponent, side, octave->slices, &octave->header);
    octave->startColumn = (octave->header.offsetX + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][0]) >> 24;
    octave->startRow = (octave->header.offsetZ + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][0]) >> 24;
}

// Gets the corner slices on either side of a row of samples, and the row's z fraction and fade
DEV void probeRow(const ProbeOctave &octave, int half, int lacunarityExponent, int side, int row, uint64_t *nearZ, uint64_t *farZ, float *rowFraction,
                  float *rowFade) {
    uint32_t noise = octave.header.offsetZ + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][row];
    int rowCell = (int) (uint8_t) ((noise >> 24) - octave.startRow);
    *rowFraction = fractionOf(noise);
    *rowFade = fade(*rowFraction);
    *nearZ = octave.slices[0];
    *farZ = octave.slices[1];
#pragma unroll
    for (int z = 1; z < side; z++) {
        *nearZ = rowCell == z ? octave.slices[z] : *nearZ;
        *farZ = rowCell == z ? octave.slices[z + 1] : *farZ;
    }
}

// Samples a probe octave at one point of the row (nearZ and farZ from probeRow)
DEV float sampleProbeOctave(const ProbeOctave &octave, int half, int lacunarityExponent, int column, uint64_t nearZ, uint64_t farZ, float rowFraction,
                            float rowFade) {
    uint32_t noise = octave.header.offsetX + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][column];
    int shift = 8 * (int) (uint8_t) ((noise >> 24) - octave.startColumn);
    float fraction = fractionOf(noise);
    return sampleCorners(nearZ >> shift, farZ >> shift, fraction, octave.header.yFraction, rowFraction, fade(fraction), octave.header.yFade, rowFade);
}

/**
 * @brief This kernel checks the probe score for the seeds that passed the cell score. It shuffles the lowest two octaves of
 * continentalness and erosion in the shared tables and samples the grid with them
 * 
 * @param seeds The seeds to check
 * @param probeScores The start of the probe score for each seed
 * @param seedCount How many seeds
 * @param capacity Room in outputSeeds
 * @param outputCount The number of seeds that passed
 * @param outputSeeds Where the seeds that pass go
 */
__global__ void __launch_bounds__(TABLE_THREADS, 2) // 2 blocks on an SM at once
probeKernel(const uint64_t *seeds, const double *probeScores, int seedCount, uint32_t capacity, uint32_t *outputCount, uint64_t *outputSeeds) {
    __shared__ uint32_t tableWords[64 * TABLE_THREADS];
    int blockFirst = blockIdx.x * TABLE_THREADS;
    if (blockFirst >= seedCount) {
        return;
    }
    int i = blockFirst + threadIdx.x;
    bool hasSeed = i < seedCount;
    uint64_t seed = seeds[hasSeed ? i : blockFirst];
    float score = hasSeed ? (float) probeScores[i] : 0.0f;
    uint8_t *table = tableStart(tableWords);

    // Go through continentalness and erosion
#pragma unroll 1
    for (int climateIndex = 3; climateIndex <= LAST_PROBE_CLIMATE; climateIndex++) {
        const ClimateNoise &noise = CLIMATE_NOISE[climateIndex];
        uint64_t halfLow[2], halfHigh[2];
        climateHalves(seed, noise.saltLow, noise.saltHigh, halfLow, halfHigh);

        // Both start at octave -9 and their first two amplitudes aren't 0
        const int saltIndex = 12 + noise.firstOctave;
        const float firstAmplitude = (float) (noise.amplitudes[0] * PERSISTENCE_START[noise.amplitudeCount]);
        const float secondAmplitude = (float) (noise.amplitudes[1] * PERSISTENCE_START[noise.amplitudeCount] * 0.5);
        ProbeOctave lowest[2], second[2];
#pragma unroll
        for (int half = 0; half < 2; half++) {
            buildProbeOctave(tableWords, table, halfLow[half], halfHigh[half], saltIndex, half, 9, 4, &lowest[half]);
        }
#pragma unroll
        for (int half = 0; half < 2; half++) {
            buildProbeOctave(tableWords, table, halfLow[half], halfHigh[half], saltIndex + 1, half, 8, 6, &second[half]);
        }

        float mean = 0, squareSum = 0;
        float lowestValue = 1e9f, highestValue = -1e9f;
        uint64_t binsLow = 0, binsHigh = 0;
        for (int row = 0; row < 9; row++) {
            uint64_t lowestNear[2], lowestFar[2], secondNear[2], secondFar[2];
            float lowestFraction[2], lowestFade[2], secondFraction[2], secondFade[2];
#pragma unroll
            for (int half = 0; half < 2; half++) {
                probeRow(lowest[half], half, 9, 4, row, &lowestNear[half], &lowestFar[half], &lowestFraction[half], &lowestFade[half]);
                probeRow(second[half], half, 8, 6, row, &secondNear[half], &secondFar[half], &secondFraction[half], &secondFade[half]);
            }

#pragma unroll
            for (int column = 0; column < 9; column++) {
                float halfSums[2];
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    float sum = 0;
                    sum += firstAmplitude * sampleProbeOctave(lowest[half], half, 9, column, lowestNear[half], lowestFar[half], lowestFraction[half], lowestFade[half]);
                    sum += secondAmplitude * sampleProbeOctave(second[half], half, 8, column, secondNear[half], secondFar[half], secondFraction[half], secondFade[half]);
                    halfSums[half] = sum;
                }
                float value = (halfSums[0] + halfSums[1]) * (float) noise.amplitude;
                mean += value;
                squareSum += value * value;
                lowestValue = fminf(lowestValue, value);
                highestValue = fmaxf(highestValue, value);

                int bin = (int) ((value + 1.5f) * (16.0f / 3.0f));
                bin = min(max(bin, 0), 15);
                uint64_t binBit = 1ULL << (8 * (bin & 7));
                binsLow += bin < 8 ? binBit : 0;
                binsHigh += bin < 8 ? 0 : binBit;
            }
        }

        mean *= 1.0f / SAMPLE_COUNT;
        float variance = squareSum * (1.0f / SAMPLE_COUNT) - mean * mean;
        const int firstWeight = 128 + (climateIndex - 3) * 36; // weights for this climate value start here
        score += FLOAT_PROBE_SCORE_WEIGHTS[firstWeight] * mean + FLOAT_PROBE_SCORE_WEIGHTS[firstWeight + 1] * sqrtf(fmaxf(variance, 0.0f))
               + FLOAT_PROBE_SCORE_WEIGHTS[firstWeight + 2] * lowestValue + FLOAT_PROBE_SCORE_WEIGHTS[firstWeight + 3] * highestValue;
#pragma unroll
        for (int bin = 0; bin < 16; bin++) {
            int binCount = binAmount(binsLow, binsHigh, bin);
            score += FLOAT_PROBE_SCORE_WEIGHTS[firstWeight + 4 + bin] * shareOf(binCount) + FLOAT_PROBE_SCORE_WEIGHTS[firstWeight + 20 + bin] * rootShareOf(binCount);
        }
    }

    if (hasSeed && score >= FLOAT_PROBE_SCORE_THRESHOLD) {
        uint32_t slot = atomicAdd(outputCount, 1u);
        if (slot < capacity) {
            outputSeeds[slot] = seed;
        }
    }
}

// The cascade result for a seed
struct CascadeResult {
    double score;
    int missing;
};

// The build kernel writes the cascade octaves into the tables buffer, one thread for each seed and octave. Word w of an octave
// goes to w * TABLE_OCTAVES + octave, and the cascade kernel copies it back out.
#define TABLE_OCTAVES (OCTAVE_COUNT - SHIFT_OCTAVES) // octaves the build kernel makes for a seed, everything except shift
#define OCTAVE_WORDS ((int) (sizeof(Octave) / 4))    // The size of an octave in uint32_t words
#define BUILD_CHUNK (1 << 16) // max seeds in a build chunk

/**
 * @brief This kernel builds the cascade octaves for a chunk of seeds
 * 
 * @param seeds The seeds
 * @param seedCount How many seeds are in the chunk
 * @param tables Where the octaves go
 */
__global__ void __launch_bounds__(SHUFFLE_THREADS)
buildKernel(const uint64_t *seeds, int seedCount, uint32_t *tables) {
    __shared__ uint32_t sharedTables[64 * SHUFFLE_THREADS];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seedCount * TABLE_OCTAVES) {
        return;
    }
    uint32_t *tableWords = sharedTables + threadIdx.x;
    uint8_t *table = (uint8_t *) tableWords;

    int seedIndex = i / TABLE_OCTAVES;
    int octaveIndex = i - seedIndex * TABLE_OCTAVES;
    XoroshiroState random;
    xSetSeed(&random, seeds[seedIndex]);
    uint64_t seedLow = xNextLong(&random);
    uint64_t seedHigh = xNextLong(&random);

    // The permutation array is in shared memory, the rest of the octave goes in here
    OctaveHeader header;
    buildOctave(&header, table, tableWords, SHIFT_OCTAVES + octaveIndex, seedLow, seedHigh);
    Octave octave;
    octave.permutation[256] = TABLE_BYTE(table, 0); // the copy of the first entry at the end
    octave.yLattice = header.yLattice;
    octave.shift = header.shift;
    octave.secondHalf = header.secondHalf;
    octave.offsetX = header.offsetX;
    octave.offsetZ = header.offsetZ;
    octave.amplitude = header.amplitude;
    octave.yFraction = header.yFraction;
    octave.yFade = header.yFade;

    uint32_t *destination = tables + (size_t) seedIndex * TABLE_OCTAVES * OCTAVE_WORDS;
    const uint32_t *words = (const uint32_t *) &octave;
    // The first 64 words are the permutation array (minus the last byte), we get those from shared memory
    for (int j = 0; j < OCTAVE_WORDS; j++) {
        destination[j * TABLE_OCTAVES + octaveIndex] = j < 64 ? tableWords[j * SHUFFLE_THREADS] : words[j];
    }
}

#include "rank_score.inc"

// On levels 6 and 7 biomes found + evenness has to reach these. I got them from looking at traces, the evenness at those levels
// stayed within about 0.001 of the final score. Yes they're magic numbers. They work, leave them alone unless you know what you're
// doing (--high-value raises them)
__constant__ double DEEP_CUTOFFS[2] = {51.9020, 52.9020};

// Marks the proximity gate biomes that have a climate box close to this climate, nearBiomes is a bit mask
DEV unsigned long long markNearBiomes(const int64_t climate[6], unsigned long long nearBiomes) {
    static const int AXES[5] = {0, 1, 2, 3, 5}; // temperature, humidity, continentalness, erosion and weirdness (the boxes don't have depth)
    for (int orderIndex = 0; orderIndex < PROX_BIOMES; orderIndex++) {
        int biome = PROXIMITY_ORDER[orderIndex];
        if (nearBiomes & (1ULL << biome)) {
            continue;
        }

        long long distanceSquared = 0;
        bool inRange = true;
        #pragma unroll
        for (int axis = 0; axis < 5; axis++) {
            long long value = climate[AXES[axis]];
            long long outside = value < BIOME_BOX_MIN[orderIndex][axis] ? BIOME_BOX_MIN[orderIndex][axis] - value
                              : value > BIOME_BOX_MAX[orderIndex][axis] ? value - BIOME_BOX_MAX[orderIndex][axis] : 0;
            distanceSquared += outside * outside;
            if (distanceSquared > PROX_DISTANCE_SQ) {
                inRange = false;
                break;
            }
        }

        if (inRange) {
            nearBiomes |= (1ULL << biome);
        }
    }
    return nearBiomes;
}

// index into the 128x128 coarse map (mapX and mapZ should be even)
DEV int mapIndex(int mapX, int mapZ) {
    return (mapZ >> 1) * 128 + (mapX >> 1);
}

// Bit mask of the biomes with no samples yet
DEV unsigned long long findMissingBiomes(const unsigned *biomeAmounts) {
    unsigned long long missing = 0;
    for (int i = 0; i < BIOME_COUNT; i++) {
        if (!biomeAmounts[i]) {
            missing |= 1ULL << i;
        }
    }
    return missing;
}

/**
 * @brief Checks if we can skip a cell on level 6 or 7. We skip it when the four map
 * points around the cell are all the same biome, and the missing biomes have a
 * very different climate from that biome.
 * 
 * @param coarseMap The biome map from the coarser levels
 * @param level The cascade level
 * @param gridX x of the cell on the level grid
 * @param gridZ z of the cell on the level grid
 * @param missingBiomes The biomes that don't have any samples yet (as bits)
 * @return int The biome the cell gets added to, or -1 if the cell can't be skipped
 */
DEV int skippedBiome(const uint8_t *coarseMap, int level, int gridX, int gridZ, unsigned long long missingBiomes) {
    // Find the map points around this cell
    int mapX, mapZ;
    const int step = 2;
    if (level == 6) {
        mapX = gridX & ~1;
        mapZ = gridZ & ~1;
    } else {
        mapX = (gridX >> 2) << 1;
        mapZ = (gridZ >> 2) << 1;
    }

    if (mapX + step >= 256 || mapZ + step >= 256) {
        return -1;
    }
    uint8_t cornerBiome = coarseMap[mapIndex(mapX, mapZ)];
    bool skip = (cornerBiome != 255 && coarseMap[mapIndex(mapX + step, mapZ)] == cornerBiome
                                    && coarseMap[mapIndex(mapX, mapZ + step)] == cornerBiome
                                    && coarseMap[mapIndex(mapX + step, mapZ + step)] == cornerBiome
                                    && (missingBiomes & ~FAR_BIOMES[cornerBiome]) == 0);
    if (!skip) {
        return -1;
    }
    return cornerBiome;
}

// true if the samples came close to the boxes of every proximity gate biome
DEV bool passesProximityGate(unsigned long long nearBiomes) {
    unsigned long long wanted = 0;
    for (int orderIndex = 0; orderIndex < PROX_BIOMES; orderIndex++) {
        wanted |= 1ULL << PROXIMITY_ORDER[orderIndex];
    }
    return (nearBiomes & wanted) == wanted;
}

// Float copies of the rank weights and cutoffs, uploadFloatTables fills them
__constant__ float FLOAT_RANK_SHARE_WEIGHTS[5][BIOME_COUNT];
__constant__ float FLOAT_RANK_ROOT_WEIGHTS[5][BIOME_COUNT];
__constant__ float FLOAT_RANK_FOUND_WEIGHT[5];
__constant__ float FLOAT_RANK_EVENNESS_WEIGHT[5];
__constant__ float FLOAT_RANK_SINGLES_WEIGHT[5];
__constant__ float FLOAT_RANK_BIAS[5];
__constant__ float FLOAT_RANK_THRESHOLD[5];
__constant__ float FLOAT_DEEP_CUTOFFS[2];

// Adds up a value over the warp, all the threads get the total
template <class Number>
DEV Number warpSum(Number value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

/**
 * @brief Ranks a seed after a cascade level, then returns whether or not the seed should keep going. The last level saves the score too.
 * The first warp calls this together, a thread takes two of the biomes
 * 
 * @param biomeAmounts The amount of each biome so far
 * @param level The cascade level
 * @param result Where the score gets saved on the last level
 * @return bool true if the seed should keep going
 */
DEV bool rankSeed(const unsigned *biomeAmounts, int level, CascadeResult *result) {
    int lane = threadIdx.x;
    bool hasSecond = lane + 32 < BIOME_COUNT;
    unsigned first = biomeAmounts[lane];
    unsigned second = hasSecond ? biomeAmounts[lane + 32] : 0;
    int sampleCount = warpSum((int) (first + second));
    int missing = warpSum((first == 0) + (hasSecond && second == 0));
    int singles = warpSum((first == 1) + (hasSecond && second == 1));

    float inverseCount = 1.0f / (float) sampleCount;
    float firstShare = (float) first * inverseCount;
    float secondShare = (float) second * inverseCount;
    float entropy = -warpSum((first ? firstShare * logf(firstShare) : 0.0f) + (second ? secondShare * logf(secondShare) : 0.0f));
    float evenness = entropy * (float) (1.0 / 3.9512437185814275); // log(52)
    float biomesFound = (float) (BIOME_COUNT - missing);

    bool keepGoing = true;
    if (level == LEVEL_COUNT - 1) {
        if (lane == 0) {
            result->score = missing ? 0.0 : (double) evenness;
            result->missing = missing;
        }
    } else if (level <= LAST_RANK_LEVEL) {
        const int model = level - 1;
        float terms = FLOAT_RANK_SHARE_WEIGHTS[model][lane] * firstShare + FLOAT_RANK_ROOT_WEIGHTS[model][lane] * sqrtf(firstShare);
        if (hasSecond) {
            terms += FLOAT_RANK_SHARE_WEIGHTS[model][lane + 32] * secondShare + FLOAT_RANK_ROOT_WEIGHTS[model][lane + 32] * sqrtf(secondShare);
        }
        float score = warpSum(terms) + FLOAT_RANK_BIAS[model] + FLOAT_RANK_FOUND_WEIGHT[model] * biomesFound * (1.0f / BIOME_COUNT)
                      + FLOAT_RANK_EVENNESS_WEIGHT[model] * evenness + FLOAT_RANK_SINGLES_WEIGHT[model] * (float) singles * (1.0f / BIOME_COUNT);
        keepGoing = score >= FLOAT_RANK_THRESHOLD[model];
    } else {
        keepGoing = biomesFound + evenness >= FLOAT_DEEP_CUTOFFS[level == 6 ? 0 : 1];
    }
    return keepGoing;
}

/**
 * @brief This kernel counts the biomes of a seed on finer and finer grids until the seed falls behind.
 * Seeds that make it to the end get a score. It's one block per seed.
 * 
 * @param seeds The seeds
 * @param seedCount How many seeds
 * @param lut The biome lookup table
 * @param results One result per seed
 * @param tables The octaves from the build kernel
 */
__global__ void cascadeKernel(const uint64_t *seeds, int seedCount, const uint8_t *lut,
                              CascadeResult *results, const uint32_t *tables) {
    __shared__ Octave octaves[OCTAVE_COUNT];
    __shared__ unsigned biomeAmounts[BIOME_COUNT];
    __shared__ int alive;
    if (blockIdx.x >= seedCount) {
        return;
    }

    // Levels 6 and 7 skip some cells using a map from the coarser levels. The cells that still need work go in a list, which keeps the busy threads together
    __shared__ __align__(16) uint8_t coarseMap[128 * 128];
    __shared__ unsigned long long missingBiomes;
    #define PENDING_CHUNK 4096 // levels 6 and 7 go through their cells in chunks this big
    __shared__ uint16_t pendingCells[PENDING_CHUNK];
    __shared__ int pendingCount;
    for (int i = threadIdx.x; i < 128 * 128 / 16; i += blockDim.x) {
        ((uint4 *) coarseMap)[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    }
    __shared__ unsigned long long nearBiomes;
    if (threadIdx.x == 0) {
        nearBiomes = 0;
    }
    unsigned long long threadNearBiomes = 0;

    // copy the octaves from the build kernel (shift octaves don't get built, nothing reads them)
    const uint32_t *source = tables + (size_t) blockIdx.x * TABLE_OCTAVES * OCTAVE_WORDS;
    uint32_t *destination = (uint32_t *) &octaves[SHIFT_OCTAVES];
    for (int index = threadIdx.x; index < OCTAVE_WORDS * TABLE_OCTAVES; index += blockDim.x) {
        int word = index / TABLE_OCTAVES;
        int octave = index - word * TABLE_OCTAVES;
        destination[octave * OCTAVE_WORDS + word] = source[index];
    }
    for (int i = threadIdx.x; i < BIOME_COUNT; i += blockDim.x) {
        biomeAmounts[i] = 0;
    }
    if (threadIdx.x == 0) {
        alive = 1;
        results[blockIdx.x].score = -1.0;
        results[blockIdx.x].missing = BIOME_COUNT;
    }
    __syncthreads();

    // Each level counts the cells the coarser levels didn't and adds them to the biome amounts
    for (int level = 1; level < LEVEL_COUNT; level++) {
        const bool skippingLevel = level == 6 || level == 7;
        if (skippingLevel) {
            if (threadIdx.x == 0) {
                missingBiomes = findMissingBiomes(biomeAmounts);
            }
            __syncthreads();
        }

        int stride = STRIDE[level];
        int previousStride = level > 1 ? STRIDE[level - 1] : 0;
        int cellsOnSide = 1024 / stride;
        long cellCount = skippingLevel ? 0 : (long) cellsOnSide * cellsOnSide; // 6 and 7 use the skipping pass below
        for (long i = threadIdx.x; i < cellCount; i += blockDim.x) {
            int cellX = -512 + (int) (i % cellsOnSide) * stride;
            int cellZ = -512 + (int) (i / cellsOnSide) * stride;
            if (previousStride && (cellX % previousStride) == 0 && (cellZ % previousStride) == 0) {
                continue; //a coarser level already counted this one
            }

            int64_t climate[6];
            climateAt(octaves, cellX, cellZ, climate);
            if (level <= PROX_LEVEL) {
                threadNearBiomes = markNearBiomes(climate, threadNearBiomes);
            }

            int biome = BIOME_SLOTS[lutCell(lut, climate)];
            if (biome >= 0) {
                atomicAdd(&biomeAmounts[biome], 1u);
            }
            if (level <= LAST_MAP_LEVEL) {
                // Put the biome on the coarse map
                int mapX = cellX + 512;
                int mapZ = cellZ + 512;
                coarseMap[mapIndex(mapX >> 2, mapZ >> 2)] = (uint8_t) (biome >= 0 ? biome : 255);
            }
        }

        if (skippingLevel) {
            // Go through the level in chunks. Points on the coarser grid are already counted
            const int side = (level == 6) ? 256 : 512;
            const int fineStride = (level == 6) ? 4 : 2;
            const long levelCells = (long) side * side;
            for (long firstCell = 0; firstCell < levelCells; firstCell += PENDING_CHUNK) {
                if (threadIdx.x == 0) {
                    pendingCount = 0;
                }
                __syncthreads();

                for (int j = threadIdx.x; j < PENDING_CHUNK; j += blockDim.x) {
                    long i = firstCell + j;
                    int gridX = (int) (i % side);
                    int gridZ = (int) (i / side);
                    if ((gridX & 1) == 0 && (gridZ & 1) == 0) {
                        continue;
                    }

                    int biome = skippedBiome(coarseMap, level, gridX, gridZ, missingBiomes);
                    if (biome >= 0) {
                        atomicAdd(&biomeAmounts[biome], 1u);
                    } else {
                        int position = atomicAdd(&pendingCount, 1);
                        pendingCells[position] = (uint16_t) j;
                    }
                }
                __syncthreads();

                int cellsToCheck = pendingCount;
                for (int pendingIndex = threadIdx.x; pendingIndex < cellsToCheck; pendingIndex += blockDim.x) {
                    long i = firstCell + pendingCells[pendingIndex];
                    int gridX = (int) (i % side);
                    int gridZ = (int) (i / side);
                    int cellX = -512 + gridX * fineStride;
                    int cellZ = -512 + gridZ * fineStride;
                    int64_t climate[6];
                    climateAt(octaves, cellX, cellZ, climate);
                    int biome = BIOME_SLOTS[lutCell(lut, climate)];
                    if (biome >= 0) {
                        atomicAdd(&biomeAmounts[biome], 1u);
                    }
                }
                __syncthreads();
            }
        }

        if (level <= PROX_LEVEL && threadNearBiomes) {
            atomicOr(&nearBiomes, threadNearBiomes);
        }
        __syncthreads();

        // proximity gate
        if (threadIdx.x == 0 && level == PROX_LEVEL && !passesProximityGate(nearBiomes)) {
            alive = 0;
        }

        // The first warp ranks the seed, thread 0 decides if it keeps going
        if (threadIdx.x < 32) {
            bool keepGoing = rankSeed(biomeAmounts, level, &results[blockIdx.x]);
            if (threadIdx.x == 0 && !keepGoing) {
                alive = 0;
            }
        }
        __syncthreads();

        if (!alive) {
            return;
        }
    }
}

// This part runs on the CPU

#define DEFAULT_GATE_RATE 3.5       // Percent of the stream indexes the gate lets through
#define HIGH_VALUE_GATE_RATE 7.0     // The rate high value mode uses with the gate on the GPU
#define HIGH_VALUE_CPU_GATE_RATE 10.0 // and with --cpu-gate (the CPU gate can't keep up at 7)

// The gate threshold that lets a percent of the stream indexes through. In between two of these it goes in a straight line
static const double GATE_RATES[][2] = {
    {0.25, 0.53406}, {0.50, 0.57507}, {0.75, 0.60160}, {1.00, 0.62184}, {1.25, 0.63820}, {1.50, 0.65228}, {1.75, 0.66460},
    {2.00, 0.67567}, {2.25, 0.68571}, {2.50, 0.69497}, {2.75, 0.70356}, {3.00, 0.71159}, {3.50, 0.72629}, {4.00, 0.73956},
    {4.50, 0.75174}, {5.00, 0.76305}, {6.00, 0.78361}, {7.00, 0.80214}, {8.00, 0.81922}, {9.00, 0.83518}, {10.00, 0.85029},
    {11.00, 0.86480}, {12.00, 0.87876}, {13.00, 0.89241}, {14.00, 0.90576}, {15.00, 0.91902}, {16.00, 0.93223}, {18.00, 0.95878},
    {20.00, 0.98626}, {22.00, 1.01569}, {25.00, 1.06775}};
#define GATE_RATE_COUNT ((int) (sizeof(GATE_RATES) / sizeof(GATE_RATES[0])))

static double gateThreshold = GATE_THRESHOLD; // main sets it from --gate-rate

// Looks up the gate threshold for a rate (in percent)
static double findGateThreshold(double rate) {
    for (int i = 0; i + 1 < GATE_RATE_COUNT; i++) {
        if (rate < GATE_RATES[i + 1][0]) {
            double part = (rate - GATE_RATES[i][0]) / (GATE_RATES[i + 1][0] - GATE_RATES[i][0]);
            return GATE_RATES[i][1] + part * (GATE_RATES[i + 1][1] - GATE_RATES[i][1]);
        }
    }
    return GATE_RATES[GATE_RATE_COUNT - 1][1];
}

// Runs the CPU gate on worker threads. Batches are made of whole chunks, in order, since the checkpoint counts whole chunks.
// The workers can't get more than MAX_AHEAD chunks ahead of the batches
struct GateProducer {
    enum {
        CHUNK_SIZE = 1 << 22,
        PIECE_SIZE = 1 << 16, // chunks go through the gate in pieces this big
        MAX_AHEAD = 256
    };

    uint64_t firstIndex;
    long indexCount;
    long chunkCount;

    struct Chunk {
        std::vector<uint64_t> passed; // The stream indexes that passed the gate
        long span = 0;                // chunk size before the gate
        bool ready = false;
    };
    std::vector<Chunk> chunks; // gated chunks, chunk n goes in chunks[n % MAX_AHEAD]

    long nextChunk = 0;  // The next chunk a worker will take
    long usedChunks = 0; // chunks that went into batches so far
    bool stop = false;
    long sent = 0;       // Gated indexes that went into batches
    double waitSeconds = 0; // How long nextBatch had to wait for the workers
    std::mutex lock;
    std::condition_variable chunkReady;
    std::condition_variable wantWork;
    std::vector<std::thread> workers;

    /**
     * @brief Construct a new Gate Producer object, this starts the worker threads
     * 
     * @param first The index to start from
     * @param rangeSize How many indexes to gate
     * @param threadCount The number of worker threads
     */
    GateProducer(uint64_t first, long rangeSize, int threadCount) : chunks(MAX_AHEAD) {
        firstIndex = first;
        indexCount = rangeSize;

        // round up, the last chunk can be smaller
        chunkCount = rangeSize / CHUNK_SIZE;
        if (rangeSize % CHUNK_SIZE != 0) {
            chunkCount++;
        }

        for (int i = 0; i < threadCount; i++) {
            workers.push_back(std::thread(&GateProducer::work, this));
        }
    }

    /**
     * @brief Destroy the Gate Producer object, this stops the workers and waits for them
     * 
     */
    ~GateProducer() {
        {
            std::lock_guard<std::mutex> guard(lock);
            stop = true;
        }
        wantWork.notify_all();
        for (auto &worker : workers) {
            worker.join();
        }
    }

    /**
     * @brief This method is the loop of a worker thread. It takes the next chunk, runs the gate, then hands the chunk back.
     * 
     */
    void work() {
        // sig segven
        std::vector<uint64_t> pieceOutput(PIECE_SIZE + 8);
        for (;;) {
            long chunkIndex;
            {
                std::unique_lock<std::mutex> guard(lock);
                // Wait while the workers are too far ahead of the batches
                while (!stop && nextChunk < chunkCount && nextChunk >= usedChunks + MAX_AHEAD) {
                    wantWork.wait(guard);
                }
                if (stop || nextChunk >= chunkCount) {
                    return;
                }
                chunkIndex = nextChunk;
                nextChunk++; //o'chunks reference o'chunks reference
            }

            long start = chunkIndex * (long) CHUNK_SIZE;
            long chunkSize = std::min((long) CHUNK_SIZE, indexCount - start);
            std::vector<uint64_t> passed;
            passed.reserve((size_t) (chunkSize / 16 + 64));
            for (long offset = 0; offset < chunkSize; offset += PIECE_SIZE) {
                long pieceSize = std::min((long) PIECE_SIZE, chunkSize - offset);
                size_t passCount = gateIndexes(gateThreshold, firstIndex + (uint64_t) (start + offset), (size_t) pieceSize, pieceOutput.data());
                passed.insert(passed.end(), pieceOutput.begin(), pieceOutput.begin() + (long) passCount);
            }

            {
                std::lock_guard<std::mutex> guard(lock);
                Chunk &chunk = chunks[chunkIndex % MAX_AHEAD];
                chunk.passed.swap(passed);
                chunk.span = chunkSize;
                chunk.ready = true;
            }
            chunkReady.notify_all();
        }
    }

    /**
     * @brief Fills indexes with whole chunks of gated indexes for the next batch
     * @param indexes The list to fill
     * @param span Gets set to the size of the chunks before the gate
     * @return bool false when there's nothing left
     */
    bool nextBatch(std::vector<uint64_t> &indexes, long &span) {
        indexes.clear();
        span = 0;
        std::unique_lock<std::mutex> guard(lock);
        while (usedChunks < chunkCount) {
            Chunk &chunk = chunks[usedChunks % MAX_AHEAD];
            if (!chunk.ready) {
                auto waitStart = std::chrono::steady_clock::now();
                while (!chunk.ready) {
                    chunkReady.wait(guard);
                }
                waitSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart).count();
            }

            // leave it for the next batch if it doesn't fit
            if (!indexes.empty() && (long) (indexes.size() + chunk.passed.size()) > BATCH_SIZE) {
                break;
            }

            indexes.insert(indexes.end(), chunk.passed.begin(), chunk.passed.end());
            span += chunk.span;
            chunk.passed = std::vector<uint64_t>(); // frees the memory; clear() wouldn't
            chunk.ready = false;
            usedChunks++;
            wantWork.notify_all(); // notify_one might be enough here, never tried it

            // Stop when the batch is close to full
            if ((long) indexes.size() >= BATCH_SIZE - BATCH_SIZE / 8) {
                break;
            }
        }
        sent += (long) indexes.size();
        return span > 0;
    }
};


/**
 * @brief This kernel puts the indexes the CPU threads let through at the start of a batch's index list and starts the gated
 * count there
 *
 * @param offsets The indexes from the CPU threads, minus first
 * @param count How many there are
 * @param first The first stream index of the CPU's part
 * @param indexes The batch's index list
 * @param gatedCount The batch's gated count
 */
__global__ void helperIndexesKernel(const uint32_t *offsets, uint32_t count, uint64_t first, uint64_t *indexes, uint32_t *gatedCount) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        indexes[i] = first + offsets[i];
    }
    if (i == 0) {
        *gatedCount = count;
    }
}

// The CPU threads for --cpu-assist. They gate the end of the next few batches and the GPU gate does the rest
struct GateHelper {
    enum {
        PIECE_SIZE = 1 << 20, // indexes a thread gates in one go
        LOOKAHEAD = 8         // batches ahead of the GPU the threads work on
    };

    // The end of one batch, gated by the CPU threads
    struct Part {
        long position = 0;        // where the batch starts in the range
        uint64_t first = 0;       // the first stream index of the part
        long size = 0;            // indexes in the part
        long pieceCount = 0;
        long nextPiece = 0;
        long piecesDone = 0;
        std::atomic<long> passed{0};
        uint32_t *offsets = NULL; // pinned memory, the passing indexes minus first
        long capacity = 0;
    };

    long batchSpan;
    float threshold;              // the GPU gate's cuts (times 2^24)
    float humidityThreshold;
    std::vector<Part *> parts;
    std::vector<Part *> freeParts;
    std::deque<Part *> scheduled; // in batch order
    long scheduledUpTo = 0;       // the batches that start below this have their parts
    double share = 0.3;           // the share of a batch the CPU gets
    int threadCount;
    std::atomic<long> gatedIndexes{0};
    std::atomic<long> busyNanoseconds{0};
    long usedIndexes = 0;         // Indexes the CPU gated that made it into batches
    long waits = 0;
    std::mutex lock;
    std::condition_variable workReady;
    std::condition_variable pieceDone;
    std::vector<std::thread> workers;
    bool stop = false;

    /**
     * @brief Construct a new Gate Helper object, this starts the threads
     * 
     * @param span The size of a batch before the gate
     * @param rate The gate rate in percent
     * @param threads The number of threads
     * @param streams The number of batches the GPU works on at once
     * @param gpuThreshold The GPU gate's threshold (times 2^24)
     * @param gpuHumidityThreshold The GPU gate's humidity cut (times 2^24)
     */
    GateHelper(long span, double rate, int threads, int streams, float gpuThreshold, float gpuHumidityThreshold) {
        batchSpan = span;
        threadCount = threads;
        threshold = gpuThreshold;
        humidityThreshold = gpuHumidityThreshold;
        long capacity = (long) (0.8 * span * rate / 100.0 * 1.15) + 4096; // the biggest share plus some room
        for (int i = 0; i < LOOKAHEAD + streams + 2; i++) {
            Part *part = new Part();
            if (cudaMallocHost(&part->offsets, (size_t) capacity * 4) != cudaSuccess) {
                delete part;
                break;
            }
            part->capacity = capacity;
            parts.push_back(part);
            freeParts.push_back(part);
        }
        for (int i = 0; i < threads; i++) {
            workers.push_back(std::thread(&GateHelper::work, this));
        }
    }

    /**
     * @brief Destroy the Gate Helper object, this stops the threads and waits for them
     * 
     */
    ~GateHelper() {
        {
            std::lock_guard<std::mutex> guard(lock);
            stop = true;
        }
        workReady.notify_all();
        for (auto &worker : workers) {
            worker.join();
        }
        for (Part *part : parts) {
            cudaFreeHost(part->offsets);
            delete part;
        }
    }

    /**
     * @brief This method is the loop of a thread. It takes the next piece of the oldest part that still has some, and gates it
     * 
     */
    void work() {
        std::vector<uint64_t> passedIndexes(PIECE_SIZE + 8);
        for (;;) {
            Part *part = NULL;
            long piece = 0;
            {
                std::unique_lock<std::mutex> guard(lock);
                while (!part) {
                    if (stop) {
                        return;
                    }
                    for (Part *candidate : scheduled) {
                        if (candidate->nextPiece < candidate->pieceCount) {
                            part = candidate;
                            break;
                        }
                    }
                    if (!part) {
                        workReady.wait(guard);
                    }
                }
                piece = part->nextPiece;
                part->nextPiece++;
            }

            auto started = std::chrono::steady_clock::now();
            long pieceStart = piece * (long) PIECE_SIZE;
            long pieceSize = std::min((long) PIECE_SIZE, part->size - pieceStart);
            size_t passCount = floatGateIndexes(threshold, humidityThreshold, part->first + (uint64_t) pieceStart, (size_t) pieceSize,
                                                  passedIndexes.data());
            long firstSlot = part->passed.fetch_add((long) passCount);
            if (firstSlot + (long) passCount <= part->capacity) {
                for (size_t j = 0; j < passCount; j++) {
                    part->offsets[firstSlot + (long) j] = (uint32_t) (passedIndexes[j] - part->first);
                }
            }
            busyNanoseconds += (long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count();
            gatedIndexes += pieceSize;

            {
                std::lock_guard<std::mutex> guard(lock);
                part->piecesDone++;
            }
            pieceDone.notify_all();
        }
    }

    /**
     * @brief Gives the threads the parts of the batches coming up
     * 
     * @param rangeFirst The stream index where the range starts (custom seed offset included)
     * @param queued Where the next batch starts in the range
     * @param rangeSize The size of the range
     */
    void schedule(uint64_t rangeFirst, long queued, long rangeSize) {
        std::lock_guard<std::mutex> guard(lock);
        if (scheduledUpTo < queued) {
            scheduledUpTo = queued + 2 * batchSpan; // the next two batches are too close for the threads to finish in time
        }
        while (scheduledUpTo < queued + (long) LOOKAHEAD * batchSpan && scheduledUpTo + batchSpan <= rangeSize && !freeParts.empty()) {
            Part *part = freeParts.back();
            freeParts.pop_back();
            part->position = scheduledUpTo;
            part->size = (long) (share * batchSpan);
            part->first = rangeFirst + (uint64_t) (scheduledUpTo + batchSpan - part->size);
            part->pieceCount = (part->size + PIECE_SIZE - 1) / PIECE_SIZE;
            part->nextPiece = 0;
            part->piecesDone = 0;
            part->passed = 0;
            scheduled.push_back(part);
            scheduledUpTo += batchSpan;
        }
        workReady.notify_all();
    }

    /**
     * @brief Takes the part of the batch that starts at position, when the threads are done with it
     * 
     * @param position Where the batch starts in the range
     * @return Part* The part, or NULL if that batch doesn't have one
     */
    Part *take(long position) {
        std::unique_lock<std::mutex> guard(lock);
        // Parts for batches that already went (this shouldn't happen) get dropped when the threads are done with them
        while (!scheduled.empty() && scheduled.front()->position < position) {
            Part *old = scheduled.front();
            while (old->piecesDone < old->nextPiece) {
                pieceDone.wait(guard);
            }
            old->pieceCount = 0; // so no thread picks it up again
            scheduled.pop_front();
            freeParts.push_back(old);
        }
        if (scheduled.empty() || scheduled.front()->position != position) {
            return NULL;
        }
        Part *part = scheduled.front();
        if (part->piecesDone < part->pieceCount) {
            // The threads fell behind, so the next parts get a bit smaller
            waits++;
            share *= 0.97;
            while (part->piecesDone < part->pieceCount) {
                pieceDone.wait(guard);
            }
        }
        scheduled.pop_front();
        return part;
    }

    // Gives a part's buffer back when the batch that used it is done
    void release(Part *part) {
        std::lock_guard<std::mutex> guard(lock);
        freeParts.push_back(part);
    }

    /**
     * @brief Moves the share toward what the threads can keep up with
     * 
     * @param totalRate How many indexes a second the scanner goes through
     */
    void adjust(double totalRate) {
        long busy = busyNanoseconds.load();
        if (busy <= 0 || totalRate <= 0) {
            return;
        }
        double threadRate = gatedIndexes.load() / (busy * 1e-9);
        // Aim a little under what the threads can do and move part of the way there
        double target = 0.92 * threadRate * threadCount / totalRate;
        target = std::min(0.8, std::max(0.0, target));
        std::lock_guard<std::mutex> guard(lock);
        share = 0.7 * share + 0.3 * target;
    }
};

// How far apart the GPU and CPU scores can be, since the GPU skips the coordinate warp. The GPU keeps seeds down to LOWEST_MIN_SENTS minus this
// every choice a boon or burden
#define SCORE_SLACK 2e-3

static int32_t SCORED_BIOMES[BIOME_COUNT]; // cubiomes ids of the biomes we score

// Scores from the CPU check
struct CpuScores {
    double sents;        // SENTS (Scaled ENTropy Score), our score for the 1024x1024 grid
    double arbitrations; // The ARBITRATIONS score, for the 1025x1025 grid (this grid has the edges)
    int missing;         // biomes missing from the 1024x1024 grid
};

/**
 * @brief This method scores a hit again on the CPU with cubiomes, with the coordinate warp
 * this time. It also works out the ARBITRATIONS score (the squared evenness times 100, on the grid with the edges).
 * 
 * @param seed The seed to check
 * @return CpuScores The scores from the CPU
 */
static CpuScores rescoreOnCpu(uint64_t seed) {
    BiomeNoise noise;
    initBiomeNoise(&noise, MC_26_3);
    setBiomeSeed(&noise, seed, 0);

    // slot for each biome id, -1 if we don't score it
    int8_t slotOfBiome[256];
    memset(slotOfBiome, -1, sizeof(slotOfBiome));
    for (int i = 0; i < BIOME_COUNT; i++) {
        slotOfBiome[SCORED_BIOMES[i]] = (int8_t) i;
    }

    long biomeAmounts[64], arbitrationAmounts[64];
    memset(biomeAmounts, 0, sizeof(biomeAmounts));
    memset(arbitrationAmounts, 0, sizeof(arbitrationAmounts));
    long sampleCount = 0, arbitrationSamples = 0;
    for (int z = -512; z <= 512; z++) {
        for (int x = -512; x <= 512; x++) {
            double warpedX = x + sampleDoublePerlin(&noise.climate[NP_SHIFT], x, 0, z) * 4.0;
            double warpedZ = z + sampleDoublePerlin(&noise.climate[NP_SHIFT], z, x, 0) * 4.0;
            float temperature = sampleDoublePerlin(&noise.climate[NP_TEMPERATURE], warpedX, 0, warpedZ);
            float humidity = sampleDoublePerlin(&noise.climate[NP_HUMIDITY], warpedX, 0, warpedZ);
            float continentalness = sampleDoublePerlin(&noise.climate[NP_CONTINENTALNESS], warpedX, 0, warpedZ);
            float erosion = sampleDoublePerlin(&noise.climate[NP_EROSION], warpedX, 0, warpedZ);
            float weirdness = sampleDoublePerlin(&noise.climate[NP_WEIRDNESS], warpedX, 0, warpedZ);

            int64_t climate[6];
            climate[0] = (int64_t) (10000.0F * temperature);
            climate[1] = (int64_t) (10000.0F * humidity);
            climate[2] = (int64_t) (10000.0F * continentalness);
            climate[3] = (int64_t) (10000.0F * erosion);
            climate[4] = FIXED_DEPTH;
            climate[5] = (int64_t) (10000.0F * weirdness);

            int biome = climateToBiome(MC_26_3, (const uint64_t *) climate, NULL);
            if (biome >= 0 && biome < 256 && slotOfBiome[biome] >= 0) {
                int slot = slotOfBiome[biome];
                arbitrationAmounts[slot]++;
                arbitrationSamples++;

                // Our grid doesn't have the last row and column
                if (x < 512 && z < 512) {
                    biomeAmounts[slot]++;
                    sampleCount++;
                }
            }
        }
    }

    int missing = 0;
    double entropy = 0, arbitrationEntropy = 0;
    for (int i = 0; i < BIOME_COUNT; i++) {
        if (!biomeAmounts[i]) {
            missing++;
        } else {
            double share = (double) biomeAmounts[i] / sampleCount;
            entropy -= share * log(share);
        }
        if (arbitrationAmounts[i]) {
            double share = (double) arbitrationAmounts[i] / arbitrationSamples;
            arbitrationEntropy -= share * log(share);
        }
    }

    CpuScores scores;
    scores.sents = 0.0;
    scores.missing = missing;
    if (!missing) {
        scores.sents = entropy / log((double) BIOME_COUNT);
    }

    // ARBITRATIONS doesn't go to zero when a biome is missing
    scores.arbitrations = 0.0;
    if (arbitrationSamples > 0) {
        double arbitrationEvenness = arbitrationEntropy / log((double) BIOME_COUNT);
        scores.arbitrations = arbitrationEvenness * arbitrationEvenness * 100.0;
    }
    return scores;
}

// A hit waiting on the CPU check
struct PendingHit {
    uint64_t seed;
    double gpuScore;
    int missing;
    long batchStart; // indexes that were finished before this hit's batch
    std::future<CpuScores> cpuScores;
};

// Biggest range a run can scan, keeps the chunk math in the gate from overflowing
static const long MAX_RANGE_SIZE = LONG_MAX - GateProducer::CHUNK_SIZE;

// The settings from the command line
struct Settings {
    double minSents = LOWEST_MIN_SENTS; // The SENTS score a hit needs to get saved
    double minArbitrations = 0;         // same for ARBITRATIONS
    bool prepareOnly = false;           // Check the range and save the checkpoint, but don't scan (make run does this first)
    const char *customSeed = NULL;      // text that picks the stream, NULL if it wasn't given
    bool plainStream = false;           // Scan the plain stream without a custom seed
    int gateThreads = 0;                // Threads for the CPU gate, 0 picks from the core count
    double gateRate = 0;                // Percent of the stream indexes the gate lets through, 0 if --gate-rate wasn't given
    bool highValue = false;             // tighter GPU filters for the best hits, --high-value turns them on
    int streams = STREAM_COUNT;         // Batches the GPU works on at once, fewer streams use less GPU memory
    bool cpuGate = false;               // --cpu-gate: run the gate on the CPU threads (hostgate.c)
    bool cpuAssist = false;             // --cpu-assist: the CPU threads gate part of every batch for the GPU
};

// Best seed so far, plus the hits
struct ScanResults {
    long long bestSeed = 0;     // The best hit this run saved (by SENTS)
    double bestSents = 0;
    double bestArbitrations = 0;
    long savedHits = 0;
    std::vector<PendingHit> pending;
};

/**
 * @brief Takes the hits that are done with the CPU check and saves the ones with high enough scores to the hits file
 * 
 * @param results The hits waiting for the CPU check
 * @param settings The minimum scores
 * @param waitForAll true to wait for all the checks, false to take the ones that finished
 * @param hitsFile
 */
static void saveCheckedHits(ScanResults &results, const Settings &settings, bool waitForAll, FILE *hitsFile) {
    std::vector<PendingHit> &pending = results.pending;
    size_t i = 0;
    while (i < pending.size()) {
        // skip it if the CPU check isn't done yet, unless we're waiting for all of them
        if (!waitForAll && pending[i].cpuScores.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            i++;
            continue;
        }

        CpuScores scores = pending[i].cpuScores.get();
        long long seed = (long long) pending[i].seed;
        bool verified = fabs(scores.sents - pending[i].gpuScore) < SCORE_SLACK;

        // The GPU skips the warp so sometimes it finds all the biomes when the CPU doesn't, that isn't a real mismatch
        if (!verified && scores.sents != 0.0) {
            fprintf(stderr, "The GPU and CPU don't agree on seed %lld: %.12f vs %.12f\n", seed, pending[i].gpuScore, scores.sents);
        }

        if (scores.sents >= settings.minSents && scores.arbitrations >= settings.minArbitrations) {
            // when the hit got saved, in UTC
            char foundTime[32];
            time_t now = time(NULL);
            struct tm utc;
            gmtime_r(&now, &utc);
            strftime(foundTime, sizeof(foundTime), "%Y-%m-%dT%H:%M:%SZ", &utc);

            int written = fprintf(hitsFile, "{\"seed\": %lld, \"sents\": %.9f, \"arbitrations\": %.9f, "
                                            "\"missing\": %d, \"mc\": \"26.3\", \"side\": 4096, "
                                            "\"y\": 256, \"src\": \"gpu\", \"finder\": \"JUNO v%s\", \"cpu_verified\": %s, \"found\": \"%s\"}\n",
                                  seed, scores.sents, scores.arbitrations, pending[i].missing, JUNO_VERSION, verified ? "true" : "false", foundTime);
            if (written < 0 || fflush(hitsFile) != 0) {
                fprintf(stderr, "Couldn't write seed %lld to the hits file, stopping without saving the checkpoint!\n", seed);
                exit(1);
            }
            results.savedHits++;
            if (scores.sents > results.bestSents) {
                results.bestSeed = seed;
                results.bestSents = scores.sents;
                results.bestArbitrations = scores.arbitrations;
            }
        }
        pending.erase(pending.begin() + i); // next hit moves into this spot, so no i++
    }
}

// Reads a number that's just digits (no sign or spaces). false if there's anything else in the text or it doesn't fit in 64 bits
// strtoull on its own reads 1e12 as a 1 and calls it a day, so yes, we have to babysit it
static bool readWholeNumber(const char *text, unsigned long long &value) {
    if (!isdigit((unsigned char) text[0])) {
        return false;
    }
    char *end = NULL;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0') {
        return false;
    }
    return true;
}

/**
 * @brief This method reads the checkpoint file. Starting a new range saves over this file.
 * 
 * @param path The path of the checkpoint file
 * @param offset Gets set to the stream offset from the custom seed (0 for the plain stream)
 * @param start The index the range starts at
 * @param rangeSize Size of the range
 * @param finished How many indexes are done
 * @return bool true if the checkpoint was valid
 */
static bool readCheckpoint(const char *path, uint64_t &offset, uint64_t &start, long &rangeSize, long &finished) {
    // It's all on one line, eg: juno-gpu-v1 <offset> <start> <range size> <finished>
    char tag[32] = {0};
    char fields[4][32];
    int wordsRead = 0;
    FILE *file = fopen(path, "r");
    if (file) {
        wordsRead = fscanf(file, "%31s %31s %31s %31s %31s", tag, fields[0], fields[1], fields[2], fields[3]);
        fclose(file);
    }

    unsigned long long values[4];
    bool valid = wordsRead == 5 && strcmp(tag, CHECKPOINT_TAG) == 0;
    if (valid) {
        for (int i = 0; i < 4; i++) {
            if (!readWholeNumber(fields[i], values[i])) {
                valid = false;
                break;
            }
        }
    }

    if (valid && (values[2] < 1 || values[2] > (unsigned long long) MAX_RANGE_SIZE || values[3] > values[2])) {
        valid = false;
    }

    if (!valid) {
        printf("Couldn't find a valid checkpoint in %s! Delete it, or start a new range with a start index.\n", path);
        return false;
    }

    offset = values[0];
    start = values[1];
    rangeSize = (long) values[2];
    finished = (long) values[3];
    return true;
}

/**
 * @brief Saves the checkpoint. It writes a temp file first and renames it, so if the program gets stopped in the middle the old checkpoint is still there.
 * 
 * @param path The path of the checkpoint file
 * @param offset The stream offset from the custom seed
 * @param start The index the range starts at
 * @param rangeSize Size of the range
 * @param finished The number of indexes that are finished
 * @return bool false if the checkpoint couldn't be saved
 */
static bool saveCheckpoint(const char *path, uint64_t offset, uint64_t start, long rangeSize, long finished) {
    char tempPath[512];
    snprintf(tempPath, sizeof(tempPath), "%s.tmp", path);
    FILE *file = fopen(tempPath, "w");
    bool saved = false;
    if (file) {
        saved = fprintf(file, CHECKPOINT_TAG " %llu %llu %ld %ld\n", (unsigned long long) offset, (unsigned long long) start, rangeSize, finished) > 0;
        if (fclose(file) != 0) {
            saved = false;
        }
    }
    if (saved && rename(tempPath, path) != 0) {
        saved = false;
    }
    if (!saved) {
        remove(tempPath);
        fprintf(stderr, "Couldn't save the checkpoint to %s.\n", path);
    }
    return saved;
}

// Fills SCORED_BIOMES with the surface biomes (no caves), false if there aren't BIOME_COUNT of them
static bool findScoredBiomes() {
    int biomeCount = 0;
    for (int biome = 0; biome < 256; biome++) {
        if (!biomeExists(MC_26_3, biome) || !isOverworld(MC_26_3, biome) || getDimension(biome) != DIM_OVERWORLD) {
            continue;
        }
        if (biome == deep_dark || biome == dripstone_caves || biome == lush_caves || biome == sulfur_caves) {
            continue;
        }
        if (biomeCount < BIOME_COUNT) {
            SCORED_BIOMES[biomeCount] = biome;
        }
        biomeCount++;
    }
    if (biomeCount != BIOME_COUNT) {
        printf("Invalid biome list! There should be %d biomes but we found %d\n", BIOME_COUNT, biomeCount);
        return false;
    }
    return true;
}

static void uploadBiomeSlots() {
    int8_t biomeSlots[64];
    for (int lutId = 0; lutId < 64; lutId++) {
        biomeSlots[lutId] = -1;
        for (int i = 0; i < BIOME_COUNT; i++) {
            if (lut_id[lutId] == SCORED_BIOMES[i]) {
                biomeSlots[lutId] = (int8_t) i;
            }
        }
    }
    cudaMemcpyToSymbol(BIOME_SLOTS, biomeSlots, sizeof(biomeSlots));
}

/**
 * Copies the tables for the kernel histograms to the GPU (the log weights and the humidity band edges).
 */
static void uploadHistogramTables() {
    // How many biomes have a climate box in each temperature/humidity cell. Dappled forest doesn't have a box, it counts in all of them
    const double cellWeights[25] = {14, 14, 15, 15, 13, 19, 19, 21, 20, 19, 19, 19, 19, 20, 22,
                                    24, 24, 24, 25, 26, 17, 17, 18, 18, 17};
    double bandLogWeights[5], cellLogWeights[25];
    double weightSum = 0;
    for (int band = 0; band < 5; band++) {
        bandLogWeights[band] = log(cellWeights[band * 5] + cellWeights[band * 5 + 1] + cellWeights[band * 5 + 2] + cellWeights[band * 5 + 3] + cellWeights[band * 5 + 4]);
    }
    for (int cell = 0; cell < 25; cell++) {
        cellLogWeights[cell] = log(cellWeights[cell]);
        weightSum += cellWeights[cell];
    }
    double inverseLogTotal = 1.0 / log(weightSum);

    cudaMemcpyToSymbol(BAND_LOG_WEIGHTS, bandLogWeights, sizeof(bandLogWeights));
    cudaMemcpyToSymbol(CELL_LOG_WEIGHTS, cellLogWeights, sizeof(cellLogWeights));
    cudaMemcpyToSymbol(INVERSE_LOG_TOTAL, &inverseLogTotal, sizeof(inverseLogTotal));

    // Find the lowest and highest float within 0.03 of the humidity band edges (temperature has these in HISTOGRAM_TABLE)
    float bandEdges[4], nearLow[4], nearHigh[4];
    for (int edge = 0; edge < 4; edge++) {
        double center = HUMID_EDGES[edge];
        float low = (float) center, high = (float) center;
        while (fabs((double) nextafterf(low, -10.0f) - center) < 0.03) {
            low = nextafterf(low, -10.0f);
        }
        while (fabs((double) nextafterf(high, 10.0f) - center) < 0.03) {
            high = nextafterf(high, 10.0f);
        }
        bandEdges[edge] = (float) center;
        nearLow[edge] = low;
        nearHigh[edge] = high;
    }
    cudaMemcpyToSymbol(HUMID_BAND_EDGES, bandEdges, sizeof(bandEdges));
    cudaMemcpyToSymbol(HUMID_NEAR_LOW, nearLow, sizeof(nearLow));
    cudaMemcpyToSymbol(HUMID_NEAR_HIGH, nearHigh, sizeof(nearHigh));
}

/**
 * @brief This method uploads the tighter cutoffs --high-value uses, from the temperature kernel through cascade level 7
 */
static void uploadHighValueCutoffs() {
    double temperatureEvenness = 0.980;
    double temperatureScore = 2.0;
    double cellScore = 2.0;
    double probeThreshold = 11.0;
    double rankThresholds[5];
    double deepCutoffs[2] = {51.918, 52.918};
    cudaMemcpyFromSymbol(rankThresholds, RANK_THRESHOLD, sizeof(rankThresholds)); // level 1 keeps its cutoff
    rankThresholds[1] = 16;
    rankThresholds[2] = 22;
    rankThresholds[3] = 25;
    rankThresholds[4] = 26;
    cudaMemcpyToSymbol(MIN_TEMPERATURE_EVENNESS, &temperatureEvenness, sizeof(temperatureEvenness));
    cudaMemcpyToSymbol(TEMPERATURE_SCORE_THRESHOLD, &temperatureScore, sizeof(temperatureScore));
    cudaMemcpyToSymbol(CELL_SCORE_THRESHOLD, &cellScore, sizeof(cellScore));
    cudaMemcpyToSymbol(PROBE_SCORE_THRESHOLD, &probeThreshold, sizeof(probeThreshold));
    cudaMemcpyToSymbol(RANK_THRESHOLD, rankThresholds, sizeof(rankThresholds));
    cudaMemcpyToSymbol(DEEP_CUTOFFS, deepCutoffs, sizeof(deepCutoffs));
}

// Fills HISTOGRAM_TABLE from the temperature band edges (TEMP_EDGES), the 16 bins and the near-edge windows
static void uploadTemperatureTable() {
    std::vector<uint64_t> table(HISTOGRAM_CELLS * 3);
    for (int cell = 0; cell < HISTOGRAM_CELLS; cell++) {
        double temperature = -1.0 + (cell + 0.5) / 400.0; // the middle of the grid cell
        int band = 0;
        while (band < 4 && temperature >= TEMP_EDGES[band]) {
            band++;
        }
        uint64_t word0 = 1ULL << (7 * band);
        for (int edge = 0; edge < 4; edge++) {
            if (fabs(temperature - TEMP_EDGES[edge]) < 0.03) {
                word0 += 1ULL << (7 * (5 + edge));
            }
        }
        int bin = (int) floor(8.0 * temperature) + 8;
        if (bin < 0) {
            bin = 0;
        } else if (bin > 15) {
            bin = 15;
        }
        table[3 * cell] = word0;                                                         // bands and near-edge amounts
        table[3 * cell + 1] = (bin < 8 ? 1ULL << (7 * bin) : 0) | (uint64_t) band << 56; // bins 0 to 7, the band goes up top
        table[3 * cell + 2] = bin >= 8 ? 1ULL << (7 * (bin - 8)) : 0;                    // bins 8 to 15
    }
    cudaMemcpyToSymbol(HISTOGRAM_TABLE, table.data(), table.size() * 8);
}

// Fills GRID_POSITIONS with the sample grid positions times the lacunarity (in fixed point)
static void uploadGridPositions() {
    int32_t grid[2][11][9];
    for (int half = 0; half < 2; half++) {
        for (int exponent = 0; exponent < 11; exponent++) {
            for (int step = 0; step < 9; step++) {
                double position = -512 + 128 * step;
                double noise = half == 0 ? position * ldexp(1.0, -exponent) : position * SECOND_SCALE * ldexp(1.0, -exponent);
                grid[half][exponent][step] = (int32_t) llround(noise * 16777216.0);
            }
        }
    }
    cudaMemcpyToSymbol(GRID_POSITIONS, grid, sizeof(grid));
}

// Copies a double array on the GPU into its float copy, minus the slack
template <int Size>
static void copyToFloat(const double (&symbol)[Size], const float (&floatSymbol)[Size], double slack = 0) {
    double values[Size];
    float floats[Size];
    cudaMemcpyFromSymbol(values, symbol, sizeof(values));
    for (int i = 0; i < Size; i++) {
        floats[i] = (float) (values[i] - slack);
    }
    cudaMemcpyToSymbol(floatSymbol, floats, sizeof(floats));
}

// Same thing for 2D arrays like the rank weights
template <int Rows, int Columns>
static void copyToFloat(const double (&symbol)[Rows][Columns], const float (&floatSymbol)[Rows][Columns]) {
    double values[Rows][Columns];
    float floats[Rows][Columns];
    cudaMemcpyFromSymbol(values, symbol, sizeof(values));
    for (int row = 0; row < Rows; row++) {
        for (int i = 0; i < Columns; i++) {
            floats[row][i] = (float) values[row][i];
        }
    }
    cudaMemcpyToSymbol(floatSymbol, floats, sizeof(floats));
}

static double readDouble(const double &symbol) {
    double value = 0;
    cudaMemcpyFromSymbol(&value, symbol, sizeof(value));
    return value;
}

static void writeFloat(const float &symbol, double value) {
    float single = (float) value;
    cudaMemcpyToSymbol(symbol, &single, sizeof(single));
}

/**
 * @brief This method fills the float copies of the score weights and cutoffs the kernels use. Run it after anything that changes
 * the double ones (uploadHighValueCutoffs)
 */
static void uploadFloatTables() {
    const double scoreSlack = 5e-4; // how far off the float scores can be
    copyToFloat(TEMPERATURE_SCORE_WEIGHTS, FLOAT_TEMPERATURE_SCORE_WEIGHTS);
    copyToFloat(BAND_LOG_WEIGHTS, FLOAT_BAND_LOG_WEIGHTS);
    copyToFloat(CELL_LOG_WEIGHTS, FLOAT_CELL_LOG_WEIGHTS);
    copyToFloat(CELL_SCORE_WEIGHTS, FLOAT_CELL_SCORE_WEIGHTS);
    copyToFloat(PROBE_SCORE_WEIGHTS, FLOAT_PROBE_SCORE_WEIGHTS);
    writeFloat(FLOAT_TEMPERATURE_SCORE_BIAS, readDouble(TEMPERATURE_SCORE_BIAS));
    writeFloat(FLOAT_TEMPERATURE_SCORE_THRESHOLD, readDouble(TEMPERATURE_SCORE_THRESHOLD) - scoreSlack);
    writeFloat(FLOAT_CELL_SCORE_BIAS, readDouble(CELL_SCORE_BIAS));
    writeFloat(FLOAT_CELL_SCORE_THRESHOLD, readDouble(CELL_SCORE_THRESHOLD) - scoreSlack);
    writeFloat(FLOAT_PROBE_SCORE_BIAS, readDouble(PROBE_SCORE_BIAS));
    writeFloat(FLOAT_PROBE_SCORE_THRESHOLD, readDouble(PROBE_SCORE_THRESHOLD) - scoreSlack);
    writeFloat(FLOAT_INVERSE_LOG_TOTAL, readDouble(INVERSE_LOG_TOTAL));
    double temperatureEvenness = readDouble(MIN_TEMPERATURE_EVENNESS);
    if (temperatureEvenness < MIN_CELL_EVENNESS) {
        temperatureEvenness = MIN_CELL_EVENNESS;
    }
    writeFloat(FLOAT_MIN_TEMPERATURE_EVENNESS, temperatureEvenness - FLOAT_SLACK);
    writeFloat(FLOAT_MIN_CELL_EVENNESS, MIN_CELL_EVENNESS - FLOAT_SLACK);

    copyToFloat(RANK_SHARE_WEIGHTS, FLOAT_RANK_SHARE_WEIGHTS);
    copyToFloat(RANK_ROOT_WEIGHTS, FLOAT_RANK_ROOT_WEIGHTS);
    copyToFloat(RANK_FOUND_WEIGHT, FLOAT_RANK_FOUND_WEIGHT);
    copyToFloat(RANK_EVENNESS_WEIGHT, FLOAT_RANK_EVENNESS_WEIGHT);
    copyToFloat(RANK_SINGLES_WEIGHT, FLOAT_RANK_SINGLES_WEIGHT);
    copyToFloat(RANK_BIAS, FLOAT_RANK_BIAS);
    // The rank score adds up a lot of float terms, it gets more slack
    copyToFloat(RANK_THRESHOLD, FLOAT_RANK_THRESHOLD, 1e-3);
    copyToFloat(DEEP_CUTOFFS, FLOAT_DEEP_CUTOFFS, 2e-5);
}

// A batch slot has its own stream and GPU buffers
struct BatchSlot {
    cudaStream_t stream;
    uint64_t *seeds;
    uint64_t *nextSeeds;
    uint32_t *passCount;
    CascadeResult *results;
    uint32_t *tables;
    uint64_t *records;
    double *probeScores;
    uint64_t *indexes;
    uint64_t *built;        // Where this slot's temperature and humidity build kernels save the corner slices
    uint32_t *gatedCount;   // How many indexes the GPU gate let into this batch
    uint32_t *helperOffsets; // The CPU threads' part of the batch (with --cpu-assist)
    GateHelper::Part *helperPart = NULL; // Goes back to the helper when the batch is done
    long span = 0;          // The size of the batch before the gate
    long temperatureIndexes = 0; // 0 when this slot has nothing to finish
    uint32_t survivors = 0; // seeds that went through the cascade
    bool temperatureQueued = false; // The temperature stage is queued but we haven't read its count back yet
    bool inFlight = false;
};

// Seeds that went into the humidity and probe kernels, and how many passed
struct StageTotals {
    unsigned long long humidityIn = 0;
    unsigned long long humidityPassed = 0;
    unsigned long long probeIn = 0;
    unsigned long long probePassed = 0;
};

// The lowest GPU score that goes to the CPU check, main sets it from the minimum scores
static double lowestGpuScore = LOWEST_MIN_SENTS - SCORE_SLACK;

// Reads a pass count back from the GPU, waits for the stream
static uint32_t readCount(uint32_t *deviceCount, cudaStream_t stream) {
    uint32_t passCount = 0;
    cudaMemcpyAsync(&passCount, deviceCount, 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return passCount;
}

/**
 * @brief Queues the temperature stage for a batch of gated stream indexes
 * @param batch The batch slot
 * @param hostIndexes The gated indexes
 */
static void startTemperatureStage(BatchSlot &batch, const std::vector<uint64_t> &hostIndexes) {
    long indexCount = (long) hostIndexes.size();
    batch.temperatureIndexes = indexCount;
    if (indexCount == 0) {
        return;
    }

    // Temperature kernel (saves the records), this doesn't wait for it
    cudaMemcpyAsync(batch.indexes, hostIndexes.data(), (size_t) indexCount * 8, cudaMemcpyHostToDevice, batch.stream);
    cudaMemsetAsync(batch.passCount, 0, 4, batch.stream);
    for (long first = 0; first < indexCount; first += SHUFFLE_CHUNK) {
        int chunkSize = (int) std::min((long) SHUFFLE_CHUNK, indexCount - first);
        temperatureBuildKernel<<<(chunkSize + TABLE_THREADS - 1) / TABLE_THREADS, TABLE_THREADS, 0, batch.stream>>>(batch.indexes + first, chunkSize, batch.built);
        temperatureKernel<<<(chunkSize + TEMPERATURE_THREADS - 1) / TEMPERATURE_THREADS, TEMPERATURE_THREADS, 0, batch.stream>>>(batch.indexes + first, chunkSize, batch.built,
                                                                                                               (uint32_t) RECORD_CAPACITY, batch.passCount, batch.records);
    }
}

// The float gate thresholds for the GPU gate (times 2^24), main sets them
static float gpuGateThreshold = 0, gpuGateHumidityThreshold = 0;
static bool gpuGate = true; // false with --cpu-gate

/**
 * @brief Queues the GPU gate and the temperature stage for span stream indexes, starting at first
 * @param batch The batch slot
 * @param first The first stream index (custom seed offset included)
 * @param span How many indexes
 * @param part The end of the batch the CPU threads already gated, or NULL
 */
static void startGateAndTemperature(BatchSlot &batch, uint64_t first, long span, GateHelper::Part *part = NULL) {
    batch.temperatureIndexes = span;
    long gpuSpan = span;
    cudaMemsetAsync(batch.gatedCount, 0, 4, batch.stream);
    if (part) {
        // The CPU's indexes go at the start of the list, then the GPU gate adds the rest
        uint32_t passed = (uint32_t) part->passed.load();
        gpuSpan = span - part->size;
        if (passed) {
            cudaMemcpyAsync(batch.helperOffsets, part->offsets, (size_t) passed * 4, cudaMemcpyHostToDevice, batch.stream);
        }
        helperIndexesKernel<<<passed / 256 + 1, 256, 0, batch.stream>>>(batch.helperOffsets, passed, part->first, batch.indexes, batch.gatedCount);
    }
    cudaMemsetAsync(batch.passCount, 0, 4, batch.stream);
    const long gateLaunch = 1L << 30; // a launch has to fit a uint32_t
    for (long done = 0; done < gpuSpan; done += gateLaunch) {
        uint32_t count = (uint32_t) std::min(gateLaunch, gpuSpan - done);
        gateKernel<<<(count + GATE_THREADS * GATE_ITEMS - 1) / (GATE_THREADS * GATE_ITEMS), GATE_THREADS, 0, batch.stream>>>(
            first + (uint64_t) done, count, gpuGateThreshold, gpuGateHumidityThreshold, batch.indexes, batch.gatedCount, (uint32_t) BATCH_SIZE);
    }

    // The host doesn't know how many got through the gate, the kernels read the count themselves
    for (long chunkFirst = 0; chunkFirst < BATCH_SIZE; chunkFirst += SHUFFLE_CHUNK) {
        temperatureBuildKernel<<<(SHUFFLE_CHUNK + TABLE_THREADS - 1) / TABLE_THREADS, TABLE_THREADS, 0, batch.stream>>>(
            batch.indexes + chunkFirst, SHUFFLE_CHUNK, batch.built, batch.gatedCount, (uint32_t) chunkFirst);
        temperatureKernel<<<(SHUFFLE_CHUNK + TEMPERATURE_THREADS - 1) / TEMPERATURE_THREADS, TEMPERATURE_THREADS, 0, batch.stream>>>(
            batch.indexes + chunkFirst, SHUFFLE_CHUNK, batch.built, (uint32_t) RECORD_CAPACITY, batch.passCount, batch.records, batch.gatedCount,
            (uint32_t) chunkFirst);
    }
}

/**
 * @brief This method runs the rest of the GPU stages for a batch. The cascade results stay on the GPU until completeBatch picks them up
 * @param batch The batch slot
 * @param lut The biome lookup table
 * @param stageTotals Stage totals to add to
 * @return uint32_t The number of seeds that went through the cascade
 */
static uint32_t finishBatch(BatchSlot &batch, const uint8_t *lut, StageTotals &stageTotals) {
    long indexCount = batch.temperatureIndexes;
    if (indexCount == 0) {
        return 0;
    }
    uint32_t survivors = readCount(batch.passCount, batch.stream);
    if (survivors > RECORD_CAPACITY) {
        fprintf(stderr, "The temperature record buffer is full! %u seeds passed, but there is room for %ld\n", survivors, RECORD_CAPACITY);
        survivors = (uint32_t) RECORD_CAPACITY;
    }
    //printf("%u of %ld indexes got past temperature\n", survivors, indexCount);

    // Humidity kernel, adds humidity then checks the cell score
    if (survivors) {
        cudaMemsetAsync(batch.passCount, 0, 4, batch.stream);
        for (uint32_t first = 0; first < survivors; first += SHUFFLE_CHUNK) {
            int chunkSize = (int) std::min((uint32_t) SHUFFLE_CHUNK, survivors - first);
            const uint64_t *chunkRecords = batch.records + (size_t) first * RECORD_WORDS;
            humidityBuildKernel<<<(chunkSize + TABLE_THREADS - 1) / TABLE_THREADS, TABLE_THREADS, 0, batch.stream>>>(chunkRecords, chunkSize, batch.built);
            humidityKernel<<<(chunkSize + HUMIDITY_THREADS - 1) / HUMIDITY_THREADS, HUMIDITY_THREADS, 0, batch.stream>>>(chunkRecords, chunkSize, batch.built, (uint32_t) BATCH_SIZE,
                                                                                                                    batch.passCount, batch.nextSeeds, batch.probeScores);
        }
        uint32_t kept = readCount(batch.passCount, batch.stream);
        stageTotals.humidityIn += survivors;
        stageTotals.humidityPassed += kept;
        std::swap(batch.seeds, batch.nextSeeds);
        survivors = kept;
    }

    // Probe kernel
    if (survivors) {
        cudaMemsetAsync(batch.passCount, 0, 4, batch.stream);
        probeKernel<<<(int) ((survivors + TABLE_THREADS - 1) / TABLE_THREADS), TABLE_THREADS, 0, batch.stream>>>(batch.seeds, batch.probeScores, (int) survivors, (uint32_t) BATCH_SIZE, batch.passCount, batch.nextSeeds);
        uint32_t kept = readCount(batch.passCount, batch.stream);
        stageTotals.probeIn += survivors;
        stageTotals.probePassed += kept;
        std::swap(batch.seeds, batch.nextSeeds);
        survivors = kept;
    }

    // Build the octaves then run the cascade, BUILD_CHUNK seeds at a time
    for (uint32_t offset = 0; offset < survivors; offset += BUILD_CHUNK) {
        uint32_t chunkSeeds = survivors - offset;
        if (chunkSeeds > BUILD_CHUNK) {
            chunkSeeds = BUILD_CHUNK;
        }
        buildKernel<<<(int) ((chunkSeeds * TABLE_OCTAVES + SHUFFLE_THREADS - 1) / SHUFFLE_THREADS), SHUFFLE_THREADS, 0, batch.stream>>>(batch.seeds + offset, (int) chunkSeeds, batch.tables);
        cascadeKernel<<<chunkSeeds, CASCADE_THREADS, 0, batch.stream>>>(batch.seeds + offset, (int) chunkSeeds, lut, batch.results + offset, batch.tables);
    }
    return survivors;
}

// Exits if the GPU reported an error. The checkpoint doesn't get saved, so a restart just does those batches again
static void stopOnGpuError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s, stopping without saving the checkpoint!\n", cudaGetErrorString(error));
        exit(1);
    }
}

static bool allocateBatchSlot(BatchSlot &batch) {
    if (cudaStreamCreateWithFlags(&batch.stream, cudaStreamNonBlocking) != cudaSuccess) { return false; }

    if (cudaMalloc(&batch.seeds, BATCH_SIZE * 8) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.nextSeeds, BATCH_SIZE * 8) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.passCount, 4) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.results, BATCH_SIZE * sizeof(CascadeResult)) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.tables, (size_t) BUILD_CHUNK * TABLE_OCTAVES * sizeof(Octave)) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.records, (size_t) RECORD_CAPACITY * RECORD_WORDS * 8) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.probeScores, BATCH_SIZE * 8) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.indexes, (size_t) BATCH_SIZE * 8) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.built, (size_t) SHUFFLE_CHUNK * BUILT_WORDS(6) * 8) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.gatedCount, 4) != cudaSuccess) { return false; }
    if (cudaMalloc(&batch.helperOffsets, (size_t) BATCH_SIZE * 4) != cudaSuccess) { return false; }
    return true;
}

static unsigned long long gpuGated = 0; // What the GPU gate let through this run
static GateHelper *gateHelper = NULL;   // The CPU threads helping the GPU gate (--cpu-assist)

/**
 * @brief Collects the cascade results of a finished batch and starts the CPU checks for the hits
 * @param batch The batch slot
 * @param hostResults Buffer for the cascade results on the CPU
 * @param hostSeeds Buffer for the seeds on the CPU
 * @param results Best seed and the hits
 * @param finishedBefore How many indexes were finished before this batch
 * @return long The size of the batch before the gate
 */
static long completeBatch(BatchSlot &batch, std::vector<CascadeResult> &hostResults, std::vector<uint64_t> &hostSeeds, ScanResults &results,
                          long finishedBefore) {
    cudaStreamSynchronize(batch.stream);
    if (gpuGate) {
        uint32_t gated = 0;
        cudaMemcpy(&gated, batch.gatedCount, 4, cudaMemcpyDeviceToHost);
        if (gated > BATCH_SIZE) {
            // This shouldn't happen, the batch spans leave room. If it does the extra indexes would get skipped
            fprintf(stderr, "The GPU gate let %u indexes into a batch that holds %ld, stopping without saving the checkpoint!\n", gated, BATCH_SIZE);
            exit(1);
        }
        gpuGated += gated;
    }
    uint32_t survivors = batch.survivors;
    if (survivors) {
        // sig segven
        cudaMemcpyAsync(hostResults.data(), batch.results, survivors * sizeof(CascadeResult), cudaMemcpyDeviceToHost, batch.stream);
        cudaMemcpyAsync(hostSeeds.data(), batch.seeds, survivors * 8, cudaMemcpyDeviceToHost, batch.stream);
        cudaStreamSynchronize(batch.stream);

        for (uint32_t i = 0; i < survivors; i++) {
            if (hostResults[i].score >= lowestGpuScore) {
                PendingHit hit;
                hit.seed = hostSeeds[i];
                hit.gpuScore = hostResults[i].score;
                hit.missing = hostResults[i].missing;
                hit.batchStart = finishedBefore;
                hit.cpuScores = std::async(std::launch::async, rescoreOnCpu, hit.seed);
                results.pending.push_back(std::move(hit));
            }
        }
    }
    stopOnGpuError();
    if (batch.helperPart) {
        gateHelper->release(batch.helperPart);
        batch.helperPart = NULL;
    }
    batch.inFlight = false;
    return batch.span;
}

// Where the checkpoint can be saved. A batch only counts once all of its hits are done with the CPU check,
// otherwise a crash in the meantime would skip those hits for good
static long checkpointPosition(const ScanResults &results, long finished) {
    long position = finished;
    for (size_t i = 0; i < results.pending.size(); i++) {
        if (results.pending[i].batchStart < position) {
            position = results.pending[i].batchStart;
        }
    }
    return position;
}

static float millisecondsSince(cudaEvent_t startEvent, cudaEvent_t nowEvent) {
    cudaEventRecord(nowEvent);
    cudaEventSynchronize(nowEvent);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, startEvent, nowEvent);
    return milliseconds;
}

// Saves how many indexes a second this run is doing, for make status. It doesn't matter much if this fails
static void saveSpeed(float elapsedMs, long scanned) {
    FILE *file = fopen(SPEED_PATH, "w");
    if (file) {
        fprintf(file, "%.0f\n", scanned / (elapsedMs / 1000.0));
        fclose(file);
    }
}

static void printUsage() {
    printf("Usage: scan [options] run                     Carry on from the checkpoint, or start at index 0 if there isn't one\n"
           "       scan [options] <start index> [count]   Start a new range (without a count it goes until you stop it)\n"
           "       scan score <seed> [<seed> ...]         Score seeds on the CPU (SENTS and ARBITRATIONS)\n"
           "       scan top <number> [sents|arbitrations] List the best hits in the hits file\n"
           "Options:\n"
           "  --custom-seed <text>         Scan your own stream, picked by this text (printable ASCII). Without it, a new\n"
           "                               range uses the custom seed saved in results/custom_seed.txt\n"
           "  --plain-stream               Scan the plain stream, without a custom seed\n"
           "  --min-sents <score>          The SENTS score a hit needs to get saved (%.3f if you leave it out)\n"
           "  --min-arbitrations <score>   The ARBITRATIONS score a hit needs to get saved\n"
           "  --gate-threads <n>           Threads for the CPU gate with --cpu-gate or --cpu-assist (all but one or two of the cores\n"
           "                               with --cpu-gate, and all but four with --cpu-assist, if you leave it out)\n"
           "  --streams <n>                Batches the GPU works on at once (%d if you leave it out). Fewer needs less GPU memory\n"
           "  --gate-rate <percent>        Percent of the stream indexes the gate lets through (%g if you leave it out, %g with\n"
           "                               --high-value, %g with --high-value --cpu-gate)\n"
           "  --cpu-gate                   Run the gate on the CPU threads instead of the GPU (a lot slower)\n"
           "  --cpu-assist                 The CPU threads gate part of every batch so the GPU gate has less to do (a bit faster,\n"
           "                               but it keeps the CPU busy)\n"
           "  --high-value                 Look for the best seeds (ARBITRATIONS 85 and up). The GPU filters get tighter and\n"
           "                               the gate lets more through. You get a lot fewer hits, but the good ones show up\n"
           "                               a lot faster\n"
           "  --all-hits                   The wider filters, for all the hits from SENTS %.3f up (the default)\n"
           "  --prepare                    Only check the range and save the checkpoint, without scanning (make run does this)\n",
           LOWEST_MIN_SENTS, STREAM_COUNT, DEFAULT_GATE_RATE, HIGH_VALUE_GATE_RATE, HIGH_VALUE_CPU_GATE_RATE, LOWEST_MIN_SENTS);
}

// true if the text isn't empty and it's all printable ASCII (space to ~)
static bool isPrintableAscii(const char *text) {
    if (text[0] == '\0') {
        return false;
    }

    for (int i = 0; text[i] != '\0'; i++) {
        if (text[i] < 32 || text[i] > 126) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Returns the number the stream gets moved by for a custom seed.
 * This hashes the text with FNV-1a and then mixes the hash with splitmix64, without the mix two texts that end with different letters would get streams that overlap.
 *
 * @param text The custom seed
 * @return uint64_t The stream offset
 */
static uint64_t customSeedOffset(const char *text) {
    uint64_t hash = 0xcbf29ce484222325ULL; // The FNV-1a offset basis
    for (int i = 0; text[i] != '\0'; i++) {
        hash = (hash ^ (unsigned char) text[i]) * 0x100000001b3ULL; // FNV-1a prime
    }

    // splitmix64 mix
    hash = (hash ^ (hash >> 30)) * 0xBF58476D1CE4E5B9ULL;
    hash = (hash ^ (hash >> 27)) * 0x94D049BB133111EBULL;
    return hash ^ (hash >> 31);
}

// Reads the score after --min-sents or --min-arbitrations
static bool readScore(int argc, char **argv, int optionIndex, double &score) {
    char *end = NULL;
    double value = 0;
    if (optionIndex + 1 < argc) {
        value = strtod(argv[optionIndex + 1], &end);
    }
    if (!end || end == argv[optionIndex + 1] || *end != '\0') {
        printf("Please put a score after %s.\n", argv[optionIndex]);
        return false;
    }
    score = value;
    return true;
}

/**
 * @brief Reads the options into settings and puts everything else in words
 * @param settings The settings to fill in
 * @param words The arguments that aren't options
 * @return bool false if an option is wrong
 */
static bool readOptions(int argc, char **argv, Settings &settings, std::vector<const char *> &words) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prepare") == 0) {
            settings.prepareOnly = true;
        } else if (strcmp(argv[i], "--plain-stream") == 0) {
            settings.plainStream = true;
        } else if (strcmp(argv[i], "--custom-seed") == 0) {
            if (i + 1 >= argc || !isPrintableAscii(argv[i + 1]) || strncmp(argv[i + 1], "--", 2) == 0) {
                printf("Please put some printable ASCII text after --custom-seed.\n");
                return false;
            }
            settings.customSeed = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--min-sents") == 0) {
            if (!readScore(argc, argv, i, settings.minSents)) {
                return false;
            }
            i++;
        } else if (strcmp(argv[i], "--gate-threads") == 0) {
            unsigned long long threads = 0;
            if (i + 1 >= argc || !readWholeNumber(argv[i + 1], threads) || threads < 1 || threads > 1024) {
                printf("--gate-threads has to be a whole number from 1 to 1024.\n");
                return false;
            }
            settings.gateThreads = (int) threads;
            i++;
        } else if (strcmp(argv[i], "--streams") == 0) {
            unsigned long long streams = 0;
            if (i + 1 >= argc || !readWholeNumber(argv[i + 1], streams) || streams < 1 || streams > 16) {
                printf("--streams has to be a whole number from 1 to 16.\n");
                return false;
            }
            settings.streams = (int) streams;
            i++;
        } else if (strcmp(argv[i], "--gate-rate") == 0) {
            if (!readScore(argc, argv, i, settings.gateRate)) {
                return false;
            }
            if (!(settings.gateRate >= GATE_RATES[0][0] && settings.gateRate <= GATE_RATES[GATE_RATE_COUNT - 1][0])) { // nan fails these too
                printf("--gate-rate has to be a percent from %g to %g.\n", GATE_RATES[0][0], GATE_RATES[GATE_RATE_COUNT - 1][0]);
                return false;
            }
            i++;
        } else if (strcmp(argv[i], "--high-value") == 0) {
            settings.highValue = true;
        } else if (strcmp(argv[i], "--all-hits") == 0) {
            settings.highValue = false;
        } else if (strcmp(argv[i], "--cpu-gate") == 0) {
            settings.cpuGate = true;
        } else if (strcmp(argv[i], "--cpu-assist") == 0) {
            settings.cpuAssist = true;
        } else if (strcmp(argv[i], "--min-arbitrations") == 0) {
            if (!readScore(argc, argv, i, settings.minArbitrations)) {
                return false;
            }
            i++;
        } else if (strncmp(argv[i], "--", 2) == 0) {
            printf("There's no option called %s!\n", argv[i]);
            printUsage();
            return false;
        } else {
            words.push_back(argv[i]);
        }
    }

    if (settings.plainStream && settings.customSeed) {
        printf("You can't use --plain-stream and --custom-seed together!\n");
        return false;
    }
    if (settings.cpuAssist && settings.cpuGate) {
        printf("You don't need --cpu-assist with --cpu-gate, the CPU already does all of the gate!\n");
        settings.cpuAssist = false;
    }

    if (settings.gateRate == 0) {
        settings.gateRate = DEFAULT_GATE_RATE;
        if (settings.highValue) {
            settings.gateRate = settings.cpuGate ? HIGH_VALUE_CPU_GATE_RATE : HIGH_VALUE_GATE_RATE;
        }
    }

    // The GPU already threw out the lower scores before the CPU check, so a lower minimum would miss most of those anyway. nan fails these too
    if (!(settings.minSents >= LOWEST_MIN_SENTS && settings.minSents <= 1.0)) {
        printf("--min-sents has to be between %g and 1, the GPU filters are tuned for SENTS scores of %g and up.\n", LOWEST_MIN_SENTS, LOWEST_MIN_SENTS);
        return false;
    }
    if (settings.minArbitrations != 0 && !(settings.minArbitrations >= LOWEST_MIN_ARBITRATIONS && settings.minArbitrations <= 100.0)) {
        printf("--min-arbitrations has to be between %g and 100, the GPU filters are tuned for SENTS scores of %g and up.\n",
               LOWEST_MIN_ARBITRATIONS, LOWEST_MIN_SENTS);
        return false;
    }
    return true;
}

/**
 * @brief Gets the custom seed a new range uses when there isn't one on the command line. The first time
 * this picks a random one and saves it, after that it reads the saved one.
 * @param text The custom seed
 * @return bool false if the custom seed couldn't be read or saved
 */
static bool readSavedCustomSeed(std::string &text) {
    FILE *file = fopen(CUSTOM_SEED_PATH, "r");
    if (file) {
        char line[256] = {0};
        bool gotLine = fgets(line, sizeof(line), file) != NULL;
        fclose(file);

        // Take the line break off the end
        for (int i = 0; line[i] != '\0'; i++) {
            if (line[i] == '\r' || line[i] == '\n') {
                line[i] = '\0';
                break;
            }
        }

        if (!gotLine || !isPrintableAscii(line) || strncmp(line, "--", 2) == 0) {
            printf("The custom seed in %s isn't printable ASCII text.\n", CUSTOM_SEED_PATH);
            return false;
        }
        text = line;
        printf("Using the custom seed saved in %s: %s\n", CUSTOM_SEED_PATH, line);
        return true;
    }

    // Pick 16 random letters and digits
    char lettersndigits[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::random_device random;
    std::uniform_int_distribution<int> pick(0, 61); //index into letters
    text.clear();
    for (int i = 0; i < 16; i++) {
        text += lettersndigits[pick(random)];
    }

    // Save it for the next range
    mkdir(RESULTS_FOLDER, 0755);
    file = fopen(CUSTOM_SEED_PATH, "w");
    bool saved = false;
    if (file) {
        saved = fprintf(file, "%s\n", text.c_str()) > 0;
        if (fclose(file) != 0) {
            saved = false;
        }
    }
    if (!saved) {
        printf("Couldn't save the custom seed to %s!\n", CUSTOM_SEED_PATH);
        return false;
    }
    printf("Picked a random custom seed and saved it to %s: %s\n", CUSTOM_SEED_PATH, text.c_str());
    return true;
}

// true if the words are run, or a start index with maybe a count (the scanner checks the numbers later)
static bool isScanCommand(const std::vector<const char *> &words) {
    if (words.size() == 1 && strcmp(words[0], "run") == 0) {
        return true;
    }
    return (words.size() == 1 || words.size() == 2) && isdigit((unsigned char) words[0][0]);
}

/**
 * @brief This method works out the gate threshold and the lowest score the GPU keeps
 * 
 * @param settings The options the command line gave
 */
static void applySettings(const Settings &settings) {
    gateThreshold = findGateThreshold(settings.gateRate);

    // The GPU throws out hits that can't make the minimums. ARBITRATIONS is about 100 times SENTS squared, the 0.05 is some room
    double neededSents = settings.minSents;
    if (settings.minArbitrations > 0) {
        double fromArbitrations = sqrt((settings.minArbitrations - 0.05) / 100.0);
        if (fromArbitrations > neededSents) {
            neededSents = fromArbitrations;
        }
    }
    lowestGpuScore = neededSents - SCORE_SLACK;
}

/**
 * @brief Works out the range to scan from the arguments. run carries on from the checkpoint (or starts at index 0 when
 * there's no checkpoint yet), and a start index with an optional count starts a new range.
 *
 * @param words The arguments that aren't options
 * @param settings The settings (for the custom seed)
 * @param offset Gets set to the stream offset from the custom seed
 * @param start The index the range starts at
 * @param rangeSize The size of the range
 * @param finished How many indexes are already done
 * @return bool false if the arguments or the checkpoint are wrong
 */
static bool readRange(const std::vector<const char *> &words, const Settings &settings, uint64_t &offset, uint64_t &start,
                      long &rangeSize, long &finished) {
    if (settings.customSeed) {
        offset = customSeedOffset(settings.customSeed);
    }
    // ebic minecraft mod reference
    bool carryOn = words.size() == 1 && strcmp(words[0], "run") == 0;

    if (carryOn && access(CHECKPOINT_PATH, F_OK) == 0) {
        uint64_t savedOffset = 0;
        if (!readCheckpoint(CHECKPOINT_PATH, savedOffset, start, rangeSize, finished)) {
            return false;
        }
        if (settings.customSeed && savedOffset != offset) {
            printf("The checkpoint is for a different custom seed! To start a range with this one, give a start index (make run START=0 CUSTOM_SEED=...)\n");
            return false;
        }
        if (settings.plainStream && savedOffset != 0) {
            printf("The checkpoint isn't for the plain stream. To scan the plain stream, give a start index (make run START=0 OPTIONS=--plain-stream)\n");
            return false;
        }
        offset = savedOffset;

        if (finished < rangeSize) {
            printf("Carrying on from the checkpoint, %ld indexes are done so far...\n", finished);
        } else {
            start += (uint64_t) rangeSize;
            finished = 0;
            printf("The range in the checkpoint is done, starting the next one from %llu...\n", (unsigned long long) start);
        }
    } else {
        // Without a count the range is as big as it can be, so it goes until it gets stopped
        unsigned long long startValue = 0, countValue = (unsigned long long) MAX_RANGE_SIZE;
        if (carryOn) {
            printf("There's no checkpoint yet, so this starts at index 0\n");
        } else {
            if (!readWholeNumber(words[0], startValue)) {
                printf("The start index has to be a whole number from 0 to %llu.\n", ULLONG_MAX);
                return false;
            }
            if (words.size() == 2 && (!readWholeNumber(words[1], countValue) || countValue < 1 || countValue > (unsigned long long) MAX_RANGE_SIZE)) {
                printf("The count has to be a whole number from 1 to %ld.\n", MAX_RANGE_SIZE);
                return false;
            }
        }
        start = startValue;
        rangeSize = (long) countValue;
        finished = 0;

        // A new range without a custom seed uses the saved one
        if (!settings.customSeed && !settings.plainStream) {
            std::string savedSeed;
            if (!readSavedCustomSeed(savedSeed)) {
                return false;
            }
            offset = customSeedOffset(savedSeed.c_str());
        }
    }

    if (offset) {
        printf("The custom seed moves the stream by %llu\n", (unsigned long long) offset);
    }
    return true;
}

// Reads a Minecraft seed (it can be negative)
static bool readSeed(const char *text, uint64_t &seed) {
    unsigned long long value = 0;
    if (text[0] == '-') {
        if (!readWholeNumber(text + 1, value) || value > 9223372036854775808ULL) {
            return false;
        }
        seed = 0 - (uint64_t) value;
        return true;
    }
    if (!readWholeNumber(text, value)) {
        return false;
    }
    seed = value;
    return true;
}

// The score command. Scores seeds on the CPU the same way the scanner checks a hit, words is "score" and then the seeds
static bool scoreSeeds(const std::vector<const char *> &words) {
    std::vector<uint64_t> seeds;
    for (size_t i = 1; i < words.size(); i++) {
        uint64_t seed = 0;
        if (!readSeed(words[i], seed)) {
            printf("%s isn't a Minecraft seed!\n", words[i]);
            return false;
        }
        seeds.push_back(seed);
    }
    if (seeds.empty()) {
        printf("Usage: scan score <seed> [<seed> ...]\n");
        return false;
    }
    if (!findScoredBiomes()) {
        return false;
    }

    // A seed takes a few seconds, so score as many at a time as there are CPU threads
    size_t threadCount = std::thread::hardware_concurrency();
    if (threadCount < 1) {
        threadCount = 1;
    }

    for (size_t first = 0; first < seeds.size(); first += threadCount) {
        std::vector<std::future<CpuScores>> scores;
        for (size_t i = first; i < seeds.size() && i < first + threadCount; i++) {
            scores.push_back(std::async(std::launch::async, rescoreOnCpu, seeds[i]));
        }
        for (size_t i = 0; i < scores.size(); i++) {
            CpuScores seedScores = scores[i].get();
            printf("Seed %lld  SENTS %.9f  ARBITRATIONS %.6f\n", (long long) seeds[first + i], seedScores.sents, seedScores.arbitrations);
            fflush(stdout);
        }
    }
    return true;
}

// A hit from the hits file
struct SavedHit {
    long long seed;
    double sents;
    double arbitrations;
    bool verified;
};

// Sort orders for top. Highest SENTS first, then ARBITRATIONS, then the seed
static bool sortBySents(const SavedHit &a, const SavedHit &b) {
    if (a.sents != b.sents) {
        return a.sents > b.sents;
    }
    if (a.arbitrations != b.arbitrations) {
        return a.arbitrations > b.arbitrations;
    }
    return a.seed < b.seed;
}

static bool sortByArbitrations(const SavedHit &a, const SavedHit &b) {
    if (a.arbitrations != b.arbitrations) {
        return a.arbitrations > b.arbitrations;
    }
    return sortBySents(a, b);
}

/**
 * @brief Prints the best hits in the hits file, a seed that got saved more than once only shows up once
 * @return bool false if the arguments are wrong or there's no hits file
 */
static bool listTopHits(const std::vector<const char *> &words) {
    // words is "top", the number of hits, then sents or arbitrations (sents if you leave it out)
    bool validArguments = words.size() == 2 || words.size() == 3;
    bool byArbitrations = false;
    if (words.size() == 3) {
        if (strcmp(words[2], "arbitrations") == 0) {
            byArbitrations = true;
        } else if (strcmp(words[2], "sents") != 0) {
            validArguments = false;
        }
    }

    unsigned long long number = 0;
    if (validArguments && (!readWholeNumber(words[1], number) || number < 1)) {
        validArguments = false;
    }
    if (!validArguments) {
        printf("Usage: scan top <number> [sents|arbitrations]\n");
        return false;
    }

    FILE *file = fopen(HITS_PATH, "r");
    if (!file) {
        printf("Couldn't open %s, you don't have any hits yet.\n", HITS_PATH);
        return false;
    }

    std::vector<SavedHit> hits;
    long skippedLines = 0;
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        SavedHit hit;
        if (sscanf(line, "{\"seed\": %lld, \"sents\": %lf, \"arbitrations\": %lf", &hit.seed, &hit.sents, &hit.arbitrations) != 3) {
            skippedLines++;
            continue;
        }
        hit.verified = strstr(line, "\"cpu_verified\": false") == NULL;
        hits.push_back(hit);
    }
    fclose(file);

    if (byArbitrations) {
        std::sort(hits.begin(), hits.end(), sortByArbitrations);
    } else {
        std::sort(hits.begin(), hits.end(), sortBySents);
    }

    // Keep the copy of a seed that sorted first
    std::vector<SavedHit> differentHits;
    std::set<long long> seen;
    for (size_t i = 0; i < hits.size(); i++) {
        if (seen.count(hits[i].seed) > 0) {
            continue;
        }
        seen.insert(hits[i].seed);
        differentHits.push_back(hits[i]);
    }

    const char *sortedBy = "SENTS";
    if (byArbitrations) {
        sortedBy = "ARBITRATIONS";
    }
    printf("%zu different seeds in %s, sorted by %s\n", differentHits.size(), HITS_PATH, sortedBy);
    if (skippedLines == 1) {
        printf("(1 line didn't look like a hit and was skipped)\n");
    } else if (skippedLines > 1) {
        printf("(%ld lines didn't look like hits and were skipped)\n", skippedLines);
    }

    for (size_t i = 0; i < differentHits.size() && i < number; i++) {
        printf("%4zu. seed %lld  SENTS %.9f  ARBITRATIONS %.6f", i + 1, differentHits[i].seed, differentHits[i].sents, differentHits[i].arbitrations);
        if (!differentHits[i].verified) {
            printf("  (the GPU and CPU scores didn't agree)");
        }
        printf("\n");
    }
    return true;
}

static bool anyBatchInFlight(const BatchSlot *batches, int streams) {
    for (int i = 0; i < streams; i++) {
        if (batches[i].inFlight || batches[i].temperatureQueued) {
            return true;
        }
    }
    return false;
}

// Summary at the end of a run: the humidity and probe kernel numbers, how fast it went, the best seed etc.
static void printSummary(const StageTotals &stageTotals, long scanned, float runMs, const ScanResults &results) {
    double humidityPercent = 0, probePercent = 0;
    if (stageTotals.humidityIn) {
        humidityPercent = 100.0 * stageTotals.humidityPassed / stageTotals.humidityIn;
    }
    if (stageTotals.probeIn) {
        probePercent = 100.0 * stageTotals.probePassed / stageTotals.probeIn;
    }
    printf("Humidity kernel: %llu seeds went in and %llu passed (%.2f%%)\n", stageTotals.humidityIn, stageTotals.humidityPassed, humidityPercent);
    printf("Probe kernel: %llu seeds went in and %llu passed (%.2f%%)\n", stageTotals.probeIn, stageTotals.probePassed, probePercent);

    printf("Scanned %ld indexes in %.2f seconds (%.0f a second)", scanned, runMs / 1000.0, scanned / (runMs / 1000.0));
    if (results.savedHits > 0) {
        printf(", the best hit was seed %lld (SENTS %.9f, ARBITRATIONS %.6f)", results.bestSeed, results.bestSents, results.bestArbitrations);
    }
    if (results.savedHits == 1) {
        printf(", and 1 hit got saved\n");
    } else {
        printf(", and %ld hits got saved\n", results.savedHits);
    }
}

// Set when Ctrl+C or make stop asks the scanner to stop
static volatile sig_atomic_t stopRequested = 0;

static void requestStop(int signalNumber) {
    // stop being cruu-uu-uuueeelll
    stopRequested = 1;
}

/**
 * @brief Takes the lock that keeps a second scanner from starting on this computer. WSL doesn't say no when a second
 * scanner asks for more GPU memory, and two of them at once can freeze the whole computer. it's a great design, really :]
 * The lock goes away by itself when the program exits (even if it crashes).
 * @return bool false if another scanner has the lock
 */
static bool lockScanner() {
    int lockFile = open(LOCK_PATH, O_RDONLY | O_CREAT, 0666);
    if (lockFile < 0) {
        printf("Couldn't open %s, so there's no check for a second scanner\n", LOCK_PATH);
        return true;
    }
    if (flock(lockFile, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            // hard pass, no return
            printf("There's already a JUNO scanner running on this computer! Stop that one first (make stop), two at once can run out of GPU memory and freeze everything.\n");
            return false;
        }
        printf("Couldn't lock %s, so there's no check for a second scanner\n", LOCK_PATH);
    }
    return true; // lockFile stays open until the program exits
}

int main(int argc, char **argv) {
    printf("JUNO v%s\n", JUNO_VERSION);

    // Read the options, then the range to scan
    Settings settings;
    std::vector<const char *> words;
    uint64_t offset = 0; // The custom seed moves the stream by this
    uint64_t start = 0;
    long rangeSize = 0;
    long finished = 0;

    if (!readOptions(argc, argv, settings, words)) {
        return 1;
    }
    applySettings(settings);

    // score and top don't scan anything
    if (!words.empty() && strcmp(words[0], "score") == 0) {
        if (!scoreSeeds(words)) {
            return 1;
        }
        return 0;
    }
    if (!words.empty() && strcmp(words[0], "top") == 0) {
        if (!listTopHits(words)) {
            return 1;
        }
        return 0;
    }
    if (!isScanCommand(words)) {
        printUsage();
        return 1;
    }
    if (!lockScanner()) {
        return 1;
    }
    if (!readRange(words, settings, offset, start, rangeSize, finished)) {
        return 1;
    }
    if (settings.prepareOnly) {
        // The scan would stop right away if it can't write the hits, so find that out now
        mkdir(RESULTS_FOLDER, 0755);
        FILE *hitsFile = fopen(HITS_PATH, "a");
        if (!hitsFile) {
            fprintf(stderr, "Couldn't open %s, the hits can't be saved!\n", HITS_PATH);
            return 1;
        }
        fclose(hitsFile);
        if (!saveCheckpoint(CHECKPOINT_PATH, offset, start, rangeSize, finished)) {
            return 1;
        }
        return 0;
    }

    printf("Saving the hits with a SENTS score of %g or more", settings.minSents);
    if (settings.minArbitrations > 0) {
        printf(" and an ARBITRATIONS score of %g or more", settings.minArbitrations);
    }
    printf("\n");
    if (settings.highValue) {
        printf("Looking for the best seeds, the GPU filters are tighter and let very few hits under ARBITRATIONS 85 through\n");
    }

    mkdir(RESULTS_FOLDER, 0755); // does nothing if it's already there
    FILE *hitsFile = fopen(HITS_PATH, "a");
    if (!hitsFile) {
        fprintf(stderr, "Couldn't open %s, the hits can't be saved!\n", HITS_PATH);
        return 1;
    }

    // Copy the tables to the GPU
    if (!findScoredBiomes()) {
        return 1;
    }
    uploadBiomeSlots();
    uploadHistogramTables();
    if (settings.highValue) {
        uploadHighValueCutoffs();
    }
    uploadGridPositions();
    uploadFloatTables();
    uploadTemperatureTable();
    uint8_t *lut = uploadLut("lut263.bin");
    stopOnGpuError();

    // TODO: try pinned host memory for these, I never got around to it
    std::vector<std::vector<uint64_t> > hostIndexes(settings.streams);
    std::vector<uint64_t> hostSeeds(BATCH_SIZE);
    std::vector<CascadeResult> hostResults(BATCH_SIZE);
    for (int i = 0; i < settings.streams; i++) {
        hostIndexes[i].reserve(BATCH_SIZE);
    }
    const long finishedAtStart = finished;
    ScanResults results;

    double lastSave = 0;
    cudaEvent_t startEvent, nowEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&nowEvent);
    cudaEventRecord(startEvent);

    // The batches run together, each with a seperate stream and buffers.
    // They finish in the order they started so finished never counts a batch the GPU is still working on
    std::vector<BatchSlot> batches(settings.streams);
    for (int i = 0; i < settings.streams; i++) {
        if (!allocateBatchSlot(batches[i])) {
            fprintf(stderr, "Couldn't make the GPU buffers! %d streams need about %.1f GB of GPU memory, try fewer with --streams\n", settings.streams,
                    0.3 + 1.2 * settings.streams);
            return 1;
        }
    }

    long queued = finished;
    int slot = 0;
    StageTotals stageTotals;

    // Determine the number of threads to use for the CPU gate based on the hardware concurrency and user settings.
    int cores = (int) std::thread::hardware_concurrency();
    const size_t maxChecks = (size_t) std::max(2, cores); // hit checks running at once
    int gateThreads = settings.gateThreads > 0 ? settings.gateThreads
                                               : cores - (cores >= 16 ? 2 : 1);
    if (gateThreads < 1) {
        gateThreads = 1;
    }
    gpuGate = !settings.cpuGate;
    gpuGateThreshold = (float) (gateThreshold * 16777216.0);
    gpuGateHumidityThreshold = (float) (std::min(gateThreshold, GATE_HUMIDITY_LIMIT) * 16777216.0);
    // A GPU gate batch covers enough of the stream to fill about 3/4 of a batch after the gate
    const long gpuSpan = (long) (0.75 * BATCH_SIZE / (settings.gateRate / 100.0));
    GateProducer *gate = NULL;
    if (gpuGate) {
        uploadGpuGate();
        printf("The gate runs on the GPU, it lets %g%% of the indexes through\n", settings.gateRate);
        if (settings.cpuAssist) {
            int helperThreads = settings.gateThreads > 0 ? settings.gateThreads : cores - 4;
            if (helperThreads < 1) {
                helperThreads = 1;
            }
            gateHelper = new GateHelper(gpuSpan, settings.gateRate, helperThreads, settings.streams, gpuGateThreshold, gpuGateHumidityThreshold);
            if (gateHelper->parts.empty()) {
                printf("Couldn't get pinned memory for --cpu-assist, the GPU gates everything on its own\n");
                delete gateHelper;
                gateHelper = NULL;
            } else {
                gateHelper->schedule(offset + start, queued, rangeSize);
                printf("Using %d CPU threads to help the GPU gate\n", helperThreads);
            }
        }
    } else {
        gate = new GateProducer(offset + start + (uint64_t) finished, rangeSize - finished, gateThreads);
        if (gateLanes() == 8) {
            printf("Using %d threads for the CPU gate (with AVX-512), it lets %g%% of the indexes through\n", gateThreads, settings.gateRate);
        } else if (gateLanes() == 4) {
            printf("Using %d threads for the CPU gate (with AVX2), it lets %g%% of the indexes through\n", gateThreads, settings.gateRate);
        } else {
            printf("Using %d threads for the CPU gate, it lets %g%% of the indexes through. This CPU has neither AVX-512 nor AVX2, so the gate is a lot slower\n",
                   gateThreads, settings.gateRate);
        }
    }
    fflush(stdout);

    // Ctrl+C or make stop lets the batches that are already going finish, then saves the checkpoint.
    // sometimes you gotta let it all play out
    // A second Ctrl+C stops it right away, cuz at that point you must really mean it
    struct sigaction stopAction;
    memset(&stopAction, 0, sizeof(stopAction));
    stopAction.sa_handler = requestStop;
    stopAction.sa_flags = SA_RESETHAND | SA_RESTART;
    sigaction(SIGINT, &stopAction, NULL);
    sigaction(SIGTERM, &stopAction, NULL);

    bool stopping = false;
    double lastAdjust = 0;
    long queuedAtAdjust = 0;
    while (true) {
        if (stopRequested && !stopping) {
            stopping = true;
            printf("Stopping, the batches on the GPU and the hit checks finish first...\n");
            fflush(stdout);
        }

        // Stop when all the indexes are queued (or we're stopping) and the batches are finished
        bool moreBatches = queued < rangeSize && !stopping;
        if (!moreBatches && !anyBatchInFlight(batches.data(), settings.streams)) {
            break;
        }

        BatchSlot &batch = batches[slot];
        if (batch.inFlight) {
            finished += completeBatch(batch, hostResults, hostSeeds, results, finished);
            saveCheckedHits(results, settings, false, hitsFile);

            // If the CPU can't keep up with the hit checks the GPU waits for the oldest one
            while (results.pending.size() >= maxChecks) {
                results.pending.front().cpuScores.wait();
                saveCheckedHits(results, settings, false, hitsFile);
            }
        }

        // Queue this slot's temperature stage, without waiting for it
        long span = 0;
        if (moreBatches && gpuGate) {
            span = std::min(gpuSpan, rangeSize - queued);
            GateHelper::Part *part = NULL;
            if (gateHelper) {
                part = gateHelper->take(queued);
                // If the CPU's indexes didn't fit in the buffer the GPU just does all of the batch
                if (part && part->passed > part->capacity) {
                    gateHelper->release(part);
                    part = NULL;
                }
                if (part) {
                    gateHelper->usedIndexes += part->size;
                }
            }
            startGateAndTemperature(batch, offset + start + (uint64_t) queued, span, part);
            batch.helperPart = part;
            batch.span = span;
            batch.temperatureQueued = true;
            queued += span;
            if (gateHelper) {
                gateHelper->schedule(offset + start, queued, rangeSize);
            }
        } else if (moreBatches && gate->nextBatch(hostIndexes[slot], span)) {
            startTemperatureStage(batch, hostIndexes[slot]);
            batch.span = span;
            batch.temperatureQueued = true;
            queued += span;
        }

        // Then finish the slot that queued its temperature stage last time round
        BatchSlot &previous = batches[(slot + settings.streams - 1) % settings.streams];
        if (previous.temperatureQueued) {
            previous.survivors = finishBatch(previous, lut, stageTotals);
            previous.temperatureQueued = false;
            previous.inFlight = true;
            //printf("slot %d: %u seeds went to the cascade\n", slot, previous.survivors);
        }
        slot = (slot + 1) % settings.streams;

        float elapsedMs = millisecondsSince(startEvent, nowEvent);

        // Twice a second the CPU helper gets a share that fits how fast everything is going
        if (gateHelper && elapsedMs / 1000.0 - lastAdjust >= 0.5) {
            if (lastAdjust > 0) {
                gateHelper->adjust((queued - queuedAtAdjust) / (elapsedMs / 1000.0 - lastAdjust));
            }
            lastAdjust = elapsedMs / 1000.0;
            queuedAtAdjust = queued;
        }
        // Every 30 seconds save the checkpoint (and the speed)
        if (elapsedMs / 1000.0 - lastSave >= 30.0) {
            lastSave = elapsedMs / 1000.0;
            saveSpeed(elapsedMs, finished - finishedAtStart);
            if (!saveCheckpoint(CHECKPOINT_PATH, offset, start, rangeSize, checkpointPosition(results, finished))) {
                return 1;
            }
        }
    }

    float runMs = millisecondsSince(startEvent, nowEvent);
    if (gateHelper) {
        printf("The CPU threads gated %.1f%% of the indexes (they fell behind %ld times)\n",
               100.0 * gateHelper->usedIndexes / (double) std::max(1L, finished - finishedAtStart), gateHelper->waits);
        delete gateHelper;
        gateHelper = NULL;
    }
    if (gpuGate) {
        printf("The gate let %llu of %ld indexes through (%.3f%%)\n", gpuGated, finished - finishedAtStart,
               100.0 * gpuGated / (double) std::max(1L, finished - finishedAtStart));
    } else {
        printf("The CPU gate sent %ld of %ld indexes to the GPU (%.3f%%)\n", (long) gate->sent, finished - finishedAtStart,
               100.0 * gate->sent / (double) std::max(1L, finished - finishedAtStart));
        printf("The batches waited for the CPU gate %.1f%% of the time\n", runMs > 0 ? 100.0 * gate->waitSeconds / (runMs / 1000.0) : 0.0);
        delete gate;
    }

    // Wait for the last CPU checks
    saveCheckedHits(results, settings, true, hitsFile);
    stopOnGpuError();

    // Save the checkpoint one last time, otherwise a short stretch at the end wouldn't get saved as finished
    if (!saveCheckpoint(CHECKPOINT_PATH, offset, start, rangeSize, finished)) {
        return 1;
    }

    printSummary(stageTotals, finished - finishedAtStart, runMs, results);
    if (stopping) {
        printf("Stopped. The checkpoint is saved, make run carries on from here\n");
    }
    return 0;
}
