// JUNO (Just Use Noise Once), a GPU seed scanner for Minecraft 26.3.
// Seeds go through the CPU gate in hostgate.c first, then the temperature, humidity and probe kernels. The cascade kernel counts the biomes
// for whatever is left, and the CPU checks the hits again at the end.
#define JUNO_VERSION "1.0"
#define BIOME_COUNT 52                   // surface biomes we score
#define TEMPERATURE_THREADS 32           // Threads in a block for the temperature kernel
#define STREAM_COUNT 5                   // Batches the GPU works on at once, each one gets a CUDA stream. Tried a few on my machine and 5 worked best
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
#include "hostgate.h"
extern "C" {
#include "biomenoise.h"
}

__constant__ int8_t BIOME_SLOTS[64]; // Slot in the biome amounts for each lookup table id (-1 if we don't score that biome)

#define SAMPLE_COUNT 81 // The temperature, humidity and probe kernels sample a 9 by 9 grid

// A bin in a kernel histogram never has more than SAMPLE_COUNT samples,
// so the kernels can just look up the share, its square root and share * log(share). The log weights for the temperature bands and cells are here too
__constant__ double SHARES[SAMPLE_COUNT + 1];
__constant__ double ROOT_SHARES[SAMPLE_COUNT + 1];
__constant__ double SHARE_LOGS[SAMPLE_COUNT + 1];
__constant__ double BAND_LOG_WEIGHTS[5];
__constant__ double CELL_LOG_WEIGHTS[25];
__constant__ double INVERSE_LOG_TOTAL;
__constant__ double TEMP_EDGES[4] = {-0.45, -0.15, 0.2, 0.55}; // temperature band edges
__constant__ double HUMID_EDGES[4] = {-0.35, -0.1, 0.1, 0.3};  // humidity band edges
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

// The temperature kernel throws out seeds before we build humidity and saves a record for the humidity kernel. Cell evenness can't be
// higher than the temperature evenness, so MIN_CELL_EVENNESS works on temperature too
#define MIN_TEMPERATURE_EVENNESS 0.968 // temperature evenness a seed needs (if MIN_CELL_EVENNESS is lower)
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

/**
 * @brief Checks the temperature of a seed, and saves a record for the humidity kernel if the seed passes
 * 
 * @param seed
 * @param capacity How many records fit in the output
 * @param outputCount Counts the seeds that passed
 * @param records Where the record gets saved
 */
DEV void filterTemperature(uint64_t seed, uint32_t capacity, uint32_t *outputCount, uint64_t *records) {
    uint64_t tempBands[4] = {0, 0, 0, 0}; // temperature band of each sample, 3 bits per sample

    XoroshiroState random;
    xSetSeed(&random, seed);
    uint64_t seedLow = xNextLong(&random);
    uint64_t seedHigh = xNextLong(&random);

    uint8_t selectors[2][3 * 3 * 8];
    int startCellX[2];
    int startCellZ[2];
    uint64_t permutationWords[33]; // the two octaves take turns with this permutation array
    uint8_t *permutation = (uint8_t *) permutationWords;
    OctaveHeader halves[2];

    XoroshiroState climateRandom;
    climateRandom.low = seedLow ^ 0x5c7e6b29735f0d7fULL; // The temperature salt
    climateRandom.high = seedHigh ^ 0xf7d86f1bbc734988ULL;
    uint64_t firstLow = xNextLong(&climateRandom);
    uint64_t firstHigh = xNextLong(&climateRandom);
    uint64_t secondLow = xNextLong(&climateRandom);
    uint64_t secondHigh = xNextLong(&climateRandom);

    // This just uses the lowest octave of the Perlin noises
    XoroshiroState octaveRandom;
    octaveRandom.low = firstLow ^ OCTAVE_SALTS[2][0];
    octaveRandom.high = firstHigh ^ OCTAVE_SALTS[2][1];
    shuffleOctave(&halves[0], permutation, &octaveRandom, 1.5 * PERSISTENCE, 1.0 / 1024);
    startCellX[0] = (int) floor(-512 * halves[0].lacunarity + halves[0].offsetX);
    startCellZ[0] = (int) floor(-512 * halves[0].lacunarity + halves[0].offsetZ);
    cacheSelectors(permutation, halves[0].yLattice, startCellX[0], startCellZ[0], 3, selectors[0]);

    octaveRandom.low = secondLow ^ OCTAVE_SALTS[2][0];
    octaveRandom.high = secondHigh ^ OCTAVE_SALTS[2][1];
    shuffleOctave(&halves[1], permutation, &octaveRandom, 1.5 * PERSISTENCE, 1.0 / 1024);
    startCellX[1] = (int) floor(-512 * SECOND_SCALE * halves[1].lacunarity + halves[1].offsetX);
    startCellZ[1] = (int) floor(-512 * SECOND_SCALE * halves[1].lacunarity + halves[1].offsetZ);
    cacheSelectors(permutation, halves[1].yLattice, startCellX[1], startCellZ[1], 3, selectors[1]);

    uint64_t bandAmounts = 0;
    uint64_t tempBinsLow = 0;
    uint64_t tempBinsHigh = 0;
    int tempNearEdge[4] = {0, 0, 0, 0};
    double tempSum = 0;
    double tempSquareSum = 0;

    int sample = 0;
    for (int z = -512; z <= 512; z += 128) {
        for (int x = -512; x <= 512; x += 128, sample++) {
            double temperature = (halves[0].amplitude * samplePerlinCached(&halves[0], selectors[0], 3, startCellX[0], startCellZ[0], x * halves[0].lacunarity, z * halves[0].lacunarity)
                                + halves[1].amplitude * samplePerlinCached(&halves[1], selectors[1], 3, startCellX[1], startCellZ[1], x * SECOND_SCALE * halves[1].lacunarity, z * SECOND_SCALE * halves[1].lacunarity)) * (15.0 / 12);

            int band = 0;
            while (band < 4 && temperature >= TEMP_EDGES[band]) {
                band++;
            }
            tempBands[sample / 21] |= (uint64_t) band << (3 * (sample % 21));
            bandAmounts += 1ULL << (8 * band);

            float temperatureFloat = (float) temperature;
            tempSum += temperatureFloat;
            tempSquareSum += (double) temperatureFloat * temperatureFloat;

            int bin = (int) ((temperatureFloat + 1.0) / 2.0 * 16);
            if (bin < 0) {
                bin = 0;
            } else if (bin > 15) {
                bin = 15;
            }
            uint64_t binBit = 1ULL << (8 * (bin & 7));
            tempBinsLow += bin < 8 ? binBit : 0;
            tempBinsHigh += bin < 8 ? 0 : binBit;

#pragma unroll
            for (int edge = 0; edge < 4; edge++) {
                if (fabs(temperatureFloat - TEMP_EDGES[edge]) < 0.03) {
                    tempNearEdge[edge]++;
                }
            }
        }
    }

    double entropy = 0;
#pragma unroll
    for (int band = 0; band < 5; band++) {
        int count = packedAmount(bandAmounts, band);
        entropy += SHARES[count] * BAND_LOG_WEIGHTS[band] - SHARE_LOGS[count];
    }

    double minEvenness = (double) MIN_CELL_EVENNESS - 1e-9; // Leave a little room for rounding
    if (MIN_TEMPERATURE_EVENNESS > minEvenness) {
        minEvenness = MIN_TEMPERATURE_EVENNESS;
    }
    double tempEvenness = entropy * INVERSE_LOG_TOTAL;
    if (tempEvenness < minEvenness) {
        return;
    }

    // The temperature score guesses whether or not the cell score would pass the seed, saves us building humidity for nothing
    double tempMean = tempSum / SAMPLE_COUNT;
    double tempVariance = tempSquareSum / SAMPLE_COUNT - tempMean * tempMean;
    double score = TEMPERATURE_SCORE_BIAS + TEMPERATURE_SCORE_WEIGHTS[10] * tempEvenness + TEMPERATURE_SCORE_WEIGHTS[11] * tempMean
                   + TEMPERATURE_SCORE_WEIGHTS[12] * sqrt(tempVariance > 0 ? tempVariance : 0);
#pragma unroll
    for (int i = 0; i < 5; i++) {
        int count = packedAmount(bandAmounts, i);
        score += TEMPERATURE_SCORE_WEIGHTS[i] * SHARES[count] + TEMPERATURE_SCORE_WEIGHTS[5 + i] * ROOT_SHARES[count];
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        score += TEMPERATURE_SCORE_WEIGHTS[13 + i] * SHARES[tempNearEdge[i]];
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int count = binAmount(tempBinsLow, tempBinsHigh, i);
        score += TEMPERATURE_SCORE_WEIGHTS[17 + i] * SHARES[count] + TEMPERATURE_SCORE_WEIGHTS[33 + i] * ROOT_SHARES[count];
    }

    if (score < TEMPERATURE_SCORE_THRESHOLD) {
        return;
    }

    // Hand the seed over to the humidity kernel
    uint32_t slot = atomicAdd(outputCount, 1u);
    if (slot < capacity) {
        uint64_t *output = records + (uint64_t) slot * RECORD_WORDS;
        uint64_t sumBits;
        uint64_t squareSumBits;
        memcpy(&sumBits, &tempSum, 8);
        memcpy(&squareSumBits, &tempSquareSum, 8);

        output[0] = seed;
        output[1] = tempBands[0];
        output[2] = tempBands[1];
        output[3] = tempBands[2];
        output[4] = tempBands[3];
        output[5] = bandAmounts | ((uint64_t) tempNearEdge[0] << 40) | ((uint64_t) tempNearEdge[1] << 48) | ((uint64_t) tempNearEdge[2] << 56);
        output[6] = (uint64_t) tempNearEdge[3];
        output[7] = tempBinsLow;
        output[8] = tempBinsHigh;
        output[9] = sumBits;
        output[10] = squareSumBits;
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
DEV void unpackRecord(const uint64_t *record, unsigned char sampleBands[SAMPLE_COUNT], int tempNearEdge[4], uint64_t *tempBinsLow,
                      uint64_t *tempBinsHigh, double *tempSum, double *tempSquareSum) {
    uint64_t tempBands[4];
    for (int i = 0; i < 4; i++) {
        tempBands[i] = record[1 + i];
    }
    for (int i = 0; i < SAMPLE_COUNT; i++) {
        sampleBands[i] = (unsigned char) ((tempBands[i / 21] >> (3 * (i % 21))) & 7);
    }

    tempNearEdge[0] = (int) ((record[5] >> 40) & 255);
    tempNearEdge[1] = (int) ((record[5] >> 48) & 255);
    tempNearEdge[2] = (int) ((record[5] >> 56) & 255);
    tempNearEdge[3] = (int) (record[6] & 255);
    *tempBinsLow = record[7];
    *tempBinsHigh = record[8];

    uint64_t sumBits = record[9];
    uint64_t squareSumBits = record[10];
    memcpy(tempSum, &sumBits, 8);
    memcpy(tempSquareSum, &squareSumBits, 8);
}

/**
 * @brief Builds humidity for the seed in a record, then checks the cell evenness and the cell score
 * 
 * @param record The record from the temperature kernel
 * @param capacity Size of the output buffers
 * @param outputCount The number of seeds that passed
 * @param outputSeeds Where the seeds that pass get written
 * @param outputProbe The start of the probe score for those seeds
 */
DEV void filterHumidity(const uint64_t *record, uint32_t capacity, uint32_t *outputCount, uint64_t *outputSeeds, double *outputProbe) {
    uint64_t seed = record[0];

    XoroshiroState random;
    xSetSeed(&random, seed);
    uint64_t seedLow = xNextLong(&random);
    uint64_t seedHigh = xNextLong(&random);

    // Humidity needs a bigger block of lattice cells than temperature
    uint8_t selectors[2][6 * 6 * 8];
    int startCellX[2], startCellZ[2];
    uint64_t permutationWords[33];
    uint8_t *permutation = (uint8_t *) permutationWords;
    OctaveHeader halves[2];
    unsigned char sampleBands[SAMPLE_COUNT];

    uint64_t tempBinsLow, tempBinsHigh;
    int tempNearEdge[4];
    double tempSum, tempSquareSum;
    unpackRecord(record, sampleBands, tempNearEdge, &tempBinsLow, &tempBinsHigh, &tempSum, &tempSquareSum);

    XoroshiroState climateRandom;
    climateRandom.low = seedLow ^ 0x81bb4d22e8dc168eULL; //humidity salt
    climateRandom.high = seedHigh ^ 0xf1c8b4bea16303cdULL;
    uint64_t firstLow = xNextLong(&climateRandom);
    uint64_t firstHigh = xNextLong(&climateRandom);
    uint64_t secondLow = xNextLong(&climateRandom);
    uint64_t secondHigh = xNextLong(&climateRandom);

    // Only the lowest octave again
    XoroshiroState octaveRandom;
    octaveRandom.low = firstLow ^ OCTAVE_SALTS[4][0];
    octaveRandom.high = firstHigh ^ OCTAVE_SALTS[4][1];
    shuffleOctave(&halves[0], permutation, &octaveRandom, 1.0 * PERSISTENCE, 1.0 / 256);
    startCellX[0] = (int) floor(-512 * halves[0].lacunarity + halves[0].offsetX);
    startCellZ[0] = (int) floor(-512 * halves[0].lacunarity + halves[0].offsetZ);
    cacheSelectors(permutation, halves[0].yLattice, startCellX[0], startCellZ[0], 6, selectors[0]);

    octaveRandom.low = secondLow ^ OCTAVE_SALTS[4][0];
    octaveRandom.high = secondHigh ^ OCTAVE_SALTS[4][1];
    shuffleOctave(&halves[1], permutation, &octaveRandom, 1.0 * PERSISTENCE, 1.0 / 256);
    startCellX[1] = (int) floor(-512 * SECOND_SCALE * halves[1].lacunarity + halves[1].offsetX);
    startCellZ[1] = (int) floor(-512 * SECOND_SCALE * halves[1].lacunarity + halves[1].offsetZ);
    cacheSelectors(permutation, halves[1].yLattice, startCellX[1], startCellZ[1], 6, selectors[1]);

    // 25 temperature/humidity cells, so that histogram takes four of these
    uint64_t cellAmounts0 = 0;
    uint64_t cellAmounts1 = 0;
    uint64_t cellAmounts2 = 0;
    uint64_t cellAmounts3 = 0;
    uint64_t humidBinsLow = 0;
    uint64_t humidBinsHigh = 0;
    int humidNearEdge[4] = {0, 0, 0, 0};
    double humidSum = 0, humidSquareSum = 0;

    int sample = 0;
    for (int z = -512; z <= 512; z += 128) {
        for (int x = -512; x <= 512; x += 128, sample++) {
            double humidity = (halves[0].amplitude * samplePerlinCached(&halves[0], selectors[0], 6, startCellX[0], startCellZ[0], x * halves[0].lacunarity, z * halves[0].lacunarity)
                             + halves[1].amplitude * samplePerlinCached(&halves[1], selectors[1], 6, startCellX[1], startCellZ[1], x * SECOND_SCALE * halves[1].lacunarity, z * SECOND_SCALE * halves[1].lacunarity)) * (10.0 / 9);

            int humidBand = 0;
            while (humidBand < 4 && humidity >= HUMID_EDGES[humidBand]) {
                humidBand++;
            }

            int cell = sampleBands[sample] * 5 + humidBand;
            int word = cell >> 3;
            uint64_t cellBit = 1ULL << (8 * (cell & 7));
            cellAmounts0 += word == 0 ? cellBit : 0;
            cellAmounts1 += word == 1 ? cellBit : 0;
            // iiiiii've gooooot question marks all around me all around
            cellAmounts2 += word == 2 ? cellBit : 0;
            cellAmounts3 += word == 3 ? cellBit : 0;

            float humidityFloat = (float) humidity;
            humidSum += humidityFloat;
            humidSquareSum += (double) humidityFloat * humidityFloat;

            int bin = (int) ((humidityFloat + 1.0) / 2.0 * 16);
            if (bin < 0) {
                bin = 0;
            } else if (bin > 15) {
                bin = 15;
            }
            uint64_t binBit = 1ULL << (8 * (bin & 7));
            humidBinsLow += bin < 8 ? binBit : 0;
            humidBinsHigh += bin < 8 ? 0 : binBit;

#pragma unroll
            for (int edge = 0; edge < 4; edge++) {
                if (fabs(humidityFloat - HUMID_EDGES[edge]) < 0.03) {
                    humidNearEdge[edge]++;
                }
            }
        }
    }

    double entropy = 0;
#pragma unroll
    for (int cell = 0; cell < 25; cell++) {
        int count = cellAmount(cellAmounts0, cellAmounts1, cellAmounts2, cellAmounts3, cell);
        entropy += SHARES[count] * CELL_LOG_WEIGHTS[cell] - SHARE_LOGS[count];
    }

    double cellEvenness = entropy * INVERSE_LOG_TOTAL;
    if (cellEvenness < MIN_CELL_EVENNESS) {
        return;
    }

    double tempMean = tempSum / SAMPLE_COUNT;
    double humidMean = humidSum / SAMPLE_COUNT;
    double tempVariance = tempSquareSum / SAMPLE_COUNT - tempMean * tempMean;
    double humidVariance = humidSquareSum / SAMPLE_COUNT - humidMean * humidMean;
    double tempDeviation = sqrt(tempVariance > 0 ? tempVariance : 0);
    double humidDeviation = sqrt(humidVariance > 0 ? humidVariance : 0);

    // Add up the cell score. The probe score starts from the same numbers so it gets added up here too
    double score = CELL_SCORE_BIAS + CELL_SCORE_WEIGHTS[50] * cellEvenness;
    double probeScore = PROBE_SCORE_BIAS + PROBE_SCORE_WEIGHTS[50] * cellEvenness;
#pragma unroll
    for (int cell = 0; cell < 25; cell++) {
        int count = cellAmount(cellAmounts0, cellAmounts1, cellAmounts2, cellAmounts3, cell);
        double share = SHARES[count];
        double rootShare = ROOT_SHARES[count];
        score += CELL_SCORE_WEIGHTS[cell] * share + CELL_SCORE_WEIGHTS[25 + cell] * rootShare;
        probeScore += PROBE_SCORE_WEIGHTS[cell] * share + PROBE_SCORE_WEIGHTS[25 + cell] * rootShare;
    }

    score += CELL_SCORE_WEIGHTS[51] * tempMean + CELL_SCORE_WEIGHTS[52] * tempDeviation + CELL_SCORE_WEIGHTS[53] * humidMean + CELL_SCORE_WEIGHTS[54] * humidDeviation;
    probeScore += PROBE_SCORE_WEIGHTS[51] * tempMean + PROBE_SCORE_WEIGHTS[52] * tempDeviation + PROBE_SCORE_WEIGHTS[53] * humidMean + PROBE_SCORE_WEIGHTS[54] * humidDeviation;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        double tempEdge = SHARES[tempNearEdge[i]];
        double humidEdge = SHARES[humidNearEdge[i]];
        score += CELL_SCORE_WEIGHTS[56 + i] * tempEdge + CELL_SCORE_WEIGHTS[60 + i] * humidEdge;
        probeScore += PROBE_SCORE_WEIGHTS[56 + i] * tempEdge + PROBE_SCORE_WEIGHTS[60 + i] * humidEdge;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int tempCount = binAmount(tempBinsLow, tempBinsHigh, i);
        int humidCount = binAmount(humidBinsLow, humidBinsHigh, i);
        double tempShare = SHARES[tempCount];
        double humidShare = SHARES[humidCount];
        double tempRoot = ROOT_SHARES[tempCount];
        double humidRoot = ROOT_SHARES[humidCount];
        score += CELL_SCORE_WEIGHTS[64 + i] * tempShare + CELL_SCORE_WEIGHTS[80 + i] * tempRoot + CELL_SCORE_WEIGHTS[96 + i] * humidShare + CELL_SCORE_WEIGHTS[112 + i] * humidRoot;
        probeScore += PROBE_SCORE_WEIGHTS[64 + i] * tempShare + PROBE_SCORE_WEIGHTS[80 + i] * tempRoot + PROBE_SCORE_WEIGHTS[96 + i] * humidShare + PROBE_SCORE_WEIGHTS[112 + i] * humidRoot;
    }

    if (score < CELL_SCORE_THRESHOLD) {
        return;
    }

    uint32_t slot = atomicAdd(outputCount, 1u);
    if (slot < capacity) {
        outputSeeds[slot] = seed;
        outputProbe[slot] = probeScore;
    }
}

/**
 * @brief This kernel checks the temperature for a batch of stream indexes
 * 
 * @param indexes The stream indexes
 * @param indexCount How many indexes there are
 * @param capacity Room in the records buffer
 * @param outputCount Counts the seeds that passed (this can go past capacity)
 * @param records Output records for the seeds that pass
 */
__global__ void
temperatureKernel(const uint64_t *indexes, int indexCount,
                  uint32_t capacity, uint32_t *outputCount, uint64_t *records) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= indexCount) {
        return;
    }
    filterTemperature(streamSeed(indexes[i]), capacity, outputCount, records);
}

#define HUMIDITY_THREADS 64

/**
 * @brief This kernel checks the humidity of the seeds that passed the temperature kernel
 * 
 * @param records The records from the temperature kernel
 * @param recordCount The number of records
 * @param capacity Size of the output buffers
 * @param outputCount Seeds that passed
 * @param outputSeeds Output for the seeds that pass
 * @param outputProbe Where the start of their probe score goes
 */
__global__ void __launch_bounds__(HUMIDITY_THREADS)
humidityKernel(const uint64_t *records, int recordCount,
               uint32_t capacity, uint32_t *outputCount, uint64_t *outputSeeds, double *outputProbe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= recordCount) {
        return;
    }
    filterHumidity(records + (uint64_t) i * RECORD_WORDS, capacity, outputCount, outputSeeds, outputProbe);
}

// The probe kernel adds continentalness and erosion onto the probe score from the humidity kernel,
// using the lowest octaves at the same sample points as the other kernels
#define PROBE_OCTAVES 2
#define LAST_PROBE_CLIMATE 4 // Erosion
#define PROBE_THREADS 64

/**
 * @brief This kernel checks the probe score for the seeds that passed the cell score
 * 
 * @param seeds The seeds to check
 * @param probeScores The start of the probe score for each seed
 * @param seedCount How many seeds
 * @param capacity Room in outputSeeds
 * @param outputCount The number of seeds that passed
 * @param outputSeeds Where the seeds that pass go
 */
__global__ void __launch_bounds__(PROBE_THREADS)
probeKernel(const uint64_t *seeds, const double *probeScores, int seedCount, uint32_t capacity, uint32_t *outputCount, uint64_t *outputSeeds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seedCount) {
        return;
    }

    uint64_t seed = seeds[i];
    XoroshiroState random;
    xSetSeed(&random, seed);
    uint64_t seedLow = xNextLong(&random);
    uint64_t seedHigh = xNextLong(&random);
    double score = probeScores[i];
    OctaveHeader header;
    uint8_t permutation[257];

    // Go through continentalness and erosion
    for (int climateIndex = 3; climateIndex <= LAST_PROBE_CLIMATE; climateIndex++) {
        const ClimateNoise &noise = CLIMATE_NOISE[climateIndex];
        int halfCount = noise.slotCount / 2;
        double firstHalf[SAMPLE_COUNT];
        double secondHalf[SAMPLE_COUNT];
        for (int sample = 0; sample < SAMPLE_COUNT; sample++) {
            firstHalf[sample] = 0;
            secondHalf[sample] = 0;
        }

        for (int octave = 0; octave < PROBE_OCTAVES; octave++) {
            buildOctave(&header, permutation, noise.firstSlot + octave, seedLow, seedHigh);
            int sample = 0;
            for (int z = -512; z <= 512; z += 128) {
                for (int x = -512; x <= 512; x += 128, sample++) {
                    firstHalf[sample] += header.amplitude * samplePerlinOctave(&header, permutation, x * header.lacunarity, z * header.lacunarity);
                }
            }

            buildOctave(&header, permutation, noise.firstSlot + halfCount + octave, seedLow, seedHigh);
            sample = 0;
            for (int z = -512; z <= 512; z += 128) {
                for (int x = -512; x <= 512; x += 128, sample++) {
                    secondHalf[sample] += header.amplitude * samplePerlinOctave(&header, permutation, x * SECOND_SCALE * header.lacunarity, z * SECOND_SCALE * header.lacunarity);
                }
            }
        }

        double mean = 0;
        double squareSum = 0;
        double lowest = 1e9, highest = -1e9;
        int histogram[16];
        for (int bin = 0; bin < 16; bin++) {
            histogram[bin] = 0;
        }

        for (int sample = 0; sample < SAMPLE_COUNT; sample++) {
            float value = (float) ((firstHalf[sample] + secondHalf[sample]) * noise.amplitude);
            mean += value;
            squareSum += (double) value * value;
            if (value < lowest) {
                lowest = value;
            }
            if (value > highest) {
                highest = value;
            }

            int bin = (int) ((value + 1.5) / 3.0 * 16);
            if (bin < 0) {
                bin = 0;
            } else if (bin > 15) {
                bin = 15;
            }
            histogram[bin]++;
        }

        mean /= SAMPLE_COUNT;
        double variance = squareSum / SAMPLE_COUNT - mean * mean;
        const int firstWeight = 128 + (climateIndex - 3) * 36; // weights for this climate value start here
        score += PROBE_SCORE_WEIGHTS[firstWeight] * mean + PROBE_SCORE_WEIGHTS[firstWeight + 1] * sqrt(variance > 0 ? variance : 0) + PROBE_SCORE_WEIGHTS[firstWeight + 2] * lowest + PROBE_SCORE_WEIGHTS[firstWeight + 3] * highest;
#pragma unroll
        for (int bin = 0; bin < 16; bin++) {
            int binCount = histogram[bin];
            score += PROBE_SCORE_WEIGHTS[firstWeight + 4 + bin] * SHARES[binCount] + PROBE_SCORE_WEIGHTS[firstWeight + 20 + bin] * ROOT_SHARES[binCount];
        }
    }

    if (score >= PROBE_SCORE_THRESHOLD) {
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
#define BUILD_THREADS 256

/**
 * @brief This kernel builds the cascade octaves for a chunk of seeds
 * 
 * @param seeds The seeds
 * @param seedCount How many seeds are in the chunk
 * @param tables Where the octaves go
 */
__global__ void buildKernel(const uint64_t *seeds, int seedCount, uint32_t *tables) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seedCount * TABLE_OCTAVES) {
        return;
    }

    int seedIndex = i / TABLE_OCTAVES;
    int octaveIndex = i - seedIndex * TABLE_OCTAVES;
    XoroshiroState random;
    xSetSeed(&random, seeds[seedIndex]);
    uint64_t seedLow = xNextLong(&random);
    uint64_t seedHigh = xNextLong(&random);

    Octave octave;
    buildOctave(&octave, octave.permutation, SHIFT_OCTAVES + octaveIndex, seedLow, seedHigh);
    uint32_t *destination = tables + (size_t) seedIndex * TABLE_OCTAVES * OCTAVE_WORDS;
    const uint32_t *words = (const uint32_t *) &octave;
    for (int j = 0; j < OCTAVE_WORDS; j++) {
        destination[j * TABLE_OCTAVES + octaveIndex] = words[j];
    }
}

#include "rank_score.inc"

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

/**
 * @brief Ranks a seed after a cascade level, then returns whether or not the seed should keep going. The last level saves the score too.
 * 
 * @param biomeAmounts The amount of each biome so far
 * @param level The cascade level
 * @param result Where the score gets saved on the last level
 * @return bool true if the seed should keep going
 */
DEV bool rankSeed(const unsigned *biomeAmounts, int level, CascadeResult *result) {
    unsigned sampleCount = 0;
    int missing = 0, singles = 0;
    for (int i = 0; i < BIOME_COUNT; i++) {
        sampleCount += biomeAmounts[i];
        if (!biomeAmounts[i]) {
            missing++;
        } else if (biomeAmounts[i] == 1) {
            singles++;
        }
    }

    // Don't bother with the logs if the biome count already decides it, evenness is always between 0 and 1
    bool keepGoing = true;
    double entropy = 0;
    double evenness = 0;
    bool needEvenness = true;
    const bool useRankScore = level <= LAST_RANK_LEVEL;
    // On levels 6 and 7 biomes found + evenness has to reach this. I got the cutoffs from looking at traces, the evenness at those levels
    // stayed within about 0.001 of the final score. Yes they're magic numbers. They work, leave them alone unless you know what you're doing
    const double rankCutoff = level == 6 ? 51.9020 : 52.9020;
    double biomesFound = (double) (BIOME_COUNT - missing);
    if (!useRankScore && level != LEVEL_COUNT - 1) {
        if (biomesFound + 1.0 < rankCutoff) {
            keepGoing = false;
            needEvenness = false;
        } else if (biomesFound >= rankCutoff) {
            needEvenness = false;
        }
    }

    if (needEvenness) {
        for (int i = 0; i < BIOME_COUNT; i++) {
            if (biomeAmounts[i]) {
                double share = (double) biomeAmounts[i] / sampleCount;
                entropy -= share * log(share);
            }
        }
        evenness = entropy / log((double) BIOME_COUNT);
    }

    if (level == LEVEL_COUNT - 1) {
        result->score = missing ? 0.0 : evenness;
        result->missing = missing;
    } else if (useRankScore) {
        const int model = level - 1;
        double score = RANK_BIAS[model] + RANK_FOUND_WEIGHT[model] * biomesFound / BIOME_COUNT + RANK_EVENNESS_WEIGHT[model] * evenness + RANK_SINGLES_WEIGHT[model] * (double) singles / BIOME_COUNT;
        for (int i = 0; i < BIOME_COUNT; i++) {
            double share = (double) biomeAmounts[i] / sampleCount;
            score += RANK_SHARE_WEIGHTS[model][i] * share + RANK_ROOT_WEIGHTS[model][i] * sqrt(share);
        }
        if (score < RANK_THRESHOLD[model]) {
            keepGoing = false;
        }
    } else if (needEvenness && biomesFound + evenness < rankCutoff) {
        keepGoing = false;
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
    __shared__ uint8_t coarseMap[128 * 128];
    __shared__ unsigned long long missingBiomes;
    #define PENDING_CHUNK 4096 // levels 6 and 7 go through their cells in chunks this big
    __shared__ uint16_t pendingCells[PENDING_CHUNK];
    __shared__ int pendingCount;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
        coarseMap[i] = 255;
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

        // Thread 0 ranks the seed and decides if it keeps going
        if (threadIdx.x == 0 && !rankSeed(biomeAmounts, level, &results[blockIdx.x])) {
            alive = 0;
        }
        __syncthreads();

        if (!alive) {
            return;
        }
    }
}

// This part runs on the CPU

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
                size_t passCount = gateIndexes(GATE_THRESHOLD, firstIndex + (uint64_t) (start + offset), (size_t) pieceSize, pieceOutput.data());
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
            while (!chunk.ready) {
                chunkReady.wait(guard);
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
    long sampleCount = 0;
    long arbitrationSamples = 0;
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
    double entropy = 0;
    double arbitrationEntropy = 0;
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
            int written = fprintf(hitsFile, "{\"seed\": %lld, \"sents\": %.9f, \"arbitrations\": %.9f, "
                                            "\"missing\": %d, \"mc\": \"26.3\", \"side\": 4096, "
                                            "\"y\": 256, \"src\": \"gpu\", \"finder\": \"JUNO v%s\", \"cpu_verified\": %s}\n",
                                  seed, scores.sents, scores.arbitrations, pending[i].missing, JUNO_VERSION, verified ? "true" : "false");
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
 * Copies the lookup tables for the kernel histograms to the GPU (shares, log weights etc).
 */
static void uploadHistogramTables() {
    // How many biomes have a climate box in each temperature/humidity cell. Dappled forest doesn't have a box, it counts in all of them
    const double cellWeights[25] = {14, 14, 15, 15, 13, 19, 19, 21, 20, 19, 19, 19, 19, 20, 22,
                                    24, 24, 24, 25, 26, 17, 17, 18, 18, 17};
    double shares[SAMPLE_COUNT + 1];
    double rootShares[SAMPLE_COUNT + 1];
    double shareLogs[SAMPLE_COUNT + 1];
    double bandLogWeights[5];
    double cellLogWeights[25];
    double weightSum = 0;
    for (int samples = 0; samples <= SAMPLE_COUNT; samples++) {
        double share = (double) samples / SAMPLE_COUNT;
        shares[samples] = share;
        rootShares[samples] = sqrt(share);
        if (samples == 0) {
            shareLogs[samples] = 0.0;
        } else {
            shareLogs[samples] = share * log(share);
        }
    }
    for (int band = 0; band < 5; band++) {
        bandLogWeights[band] = log(cellWeights[band * 5] + cellWeights[band * 5 + 1] + cellWeights[band * 5 + 2] + cellWeights[band * 5 + 3] + cellWeights[band * 5 + 4]);
    }
    for (int cell = 0; cell < 25; cell++) {
        cellLogWeights[cell] = log(cellWeights[cell]);
        weightSum += cellWeights[cell];
    }
    double inverseLogTotal = 1.0 / log(weightSum);

    cudaMemcpyToSymbol(SHARES, shares, sizeof(shares));
    cudaMemcpyToSymbol(ROOT_SHARES, rootShares, sizeof(rootShares));
    cudaMemcpyToSymbol(SHARE_LOGS, shareLogs, sizeof(shareLogs));
    cudaMemcpyToSymbol(BAND_LOG_WEIGHTS, bandLogWeights, sizeof(bandLogWeights));
    cudaMemcpyToSymbol(CELL_LOG_WEIGHTS, cellLogWeights, sizeof(cellLogWeights));
    cudaMemcpyToSymbol(INVERSE_LOG_TOTAL, &inverseLogTotal, sizeof(inverseLogTotal));
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
    long span = 0;          // The size of the batch before the gate
    uint32_t survivors = 0; // seeds that went through the cascade
    bool inFlight = false;
};

// Seeds that went into the humidity and probe kernels, and how many passed
struct StageTotals {
    unsigned long long humidityIn = 0;
    unsigned long long humidityPassed = 0;
    unsigned long long probeIn = 0;
    unsigned long long probePassed = 0;
};

// Reads a pass count back from the GPU, waits for the stream
static uint32_t readCount(uint32_t *deviceCount, cudaStream_t stream) {
    uint32_t passCount = 0;
    cudaMemcpyAsync(&passCount, deviceCount, 4, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return passCount;
}

/**
 * @brief This method runs a batch of gated stream indexes through all of the GPU stages. The cascade results stay on the GPU until completeBatch picks them up.
 * @param batch The batch slot
 * @param hostIndexes The gated indexes
 * @param lut The biome lookup table
 * @param stageTotals Stage totals to add to
 * @return uint32_t The number of seeds that went through the cascade
 */
static uint32_t runBatch(BatchSlot &batch, const std::vector<uint64_t> &hostIndexes, const uint8_t *lut, StageTotals &stageTotals) {
    long indexCount = (long) hostIndexes.size();
    if (indexCount == 0) {
        return 0;
    }

    // Temperature kernel (saves the records)
    cudaMemcpyAsync(batch.indexes, hostIndexes.data(), (size_t) indexCount * 8, cudaMemcpyHostToDevice, batch.stream);
    cudaMemsetAsync(batch.passCount, 0, 4, batch.stream);
    int blocks = (int) ((indexCount + TEMPERATURE_THREADS - 1) / TEMPERATURE_THREADS);
    temperatureKernel<<<blocks, TEMPERATURE_THREADS, 0, batch.stream>>>(batch.indexes, (int) indexCount, (uint32_t) RECORD_CAPACITY, batch.passCount, batch.records);
    uint32_t survivors = readCount(batch.passCount, batch.stream);
    if (survivors > RECORD_CAPACITY) {
        fprintf(stderr, "The temperature record buffer is full! %u seeds passed, but there is room for %ld\n", survivors, RECORD_CAPACITY);
        survivors = (uint32_t) RECORD_CAPACITY;
    }
    //printf("%u of %ld indexes got past temperature\n", survivors, indexCount);

    // Humidity kernel, adds humidity then checks the cell score
    if (survivors) {
        cudaMemsetAsync(batch.passCount, 0, 4, batch.stream);
        humidityKernel<<<(int) ((survivors + HUMIDITY_THREADS - 1) / HUMIDITY_THREADS), HUMIDITY_THREADS, 0, batch.stream>>>(batch.records, (int) survivors, (uint32_t) BATCH_SIZE, batch.passCount, batch.nextSeeds, batch.probeScores);
        uint32_t kept = readCount(batch.passCount, batch.stream);
        stageTotals.humidityIn += survivors;
        stageTotals.humidityPassed += kept;
        std::swap(batch.seeds, batch.nextSeeds);
        survivors = kept;
    }

    // Probe kernel
    if (survivors) {
        cudaMemsetAsync(batch.passCount, 0, 4, batch.stream);
        probeKernel<<<(int) ((survivors + PROBE_THREADS - 1) / PROBE_THREADS), PROBE_THREADS, 0, batch.stream>>>(batch.seeds, batch.probeScores, (int) survivors, (uint32_t) BATCH_SIZE, batch.passCount, batch.nextSeeds);
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
        buildKernel<<<(int) ((chunkSeeds * TABLE_OCTAVES + BUILD_THREADS - 1) / BUILD_THREADS), BUILD_THREADS, 0, batch.stream>>>(batch.seeds + offset, (int) chunkSeeds, batch.tables);
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
    return true;
}

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
    uint32_t survivors = batch.survivors;
    if (survivors) {
        // sig segven
        cudaMemcpyAsync(hostResults.data(), batch.results, survivors * sizeof(CascadeResult), cudaMemcpyDeviceToHost, batch.stream);
        cudaMemcpyAsync(hostSeeds.data(), batch.seeds, survivors * 8, cudaMemcpyDeviceToHost, batch.stream);
        cudaStreamSynchronize(batch.stream);

        for (uint32_t i = 0; i < survivors; i++) {
            if (hostResults[i].score >= LOWEST_MIN_SENTS - SCORE_SLACK) {
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
           "  --prepare                    Only check the range and save the checkpoint, without scanning (make run does this)\n",
           LOWEST_MIN_SENTS);
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

static bool anyBatchInFlight(const BatchSlot *batches) {
    for (int i = 0; i < STREAM_COUNT; i++) {
        if (batches[i].inFlight) {
            return true;
        }
    }
    return false;
}

// Summary at the end of a run: the humidity and probe kernel numbers, how fast it went, the best seed etc.
static void printSummary(const StageTotals &stageTotals, long scanned, float runMs, const ScanResults &results) {
    double humidityPercent = 0;
    double probePercent = 0;
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
    uint8_t *lut = uploadLut("lut263.bin");
    stopOnGpuError();

    // TODO: try pinned host memory for these, I never got around to it
    std::vector<uint64_t> hostIndexes;
    std::vector<uint64_t> hostSeeds(BATCH_SIZE);
    std::vector<CascadeResult> hostResults(BATCH_SIZE);
    hostIndexes.reserve(BATCH_SIZE);
    const long finishedAtStart = finished;
    ScanResults results;

    double lastSave = 0;
    cudaEvent_t startEvent, nowEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&nowEvent);
    cudaEventRecord(startEvent);

    // The batches run together, each with a seperate stream and buffers.
    // They finish in the order they started so finished never counts a batch the GPU is still working on
    BatchSlot batches[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        if (!allocateBatchSlot(batches[i])) {
            fprintf(stderr, "Couldn't make the GPU buffers, the scanner needs about 6 GB of GPU memory!\n");
            return 1;
        }
    }

    long queued = finished;
    int slot = 0;
    StageTotals stageTotals;

    // Leave two CPU threads for the GPU driver and the hit checks
    int gateThreads = (int) std::thread::hardware_concurrency() - 2;
    if (gateThreads < 1) {
        gateThreads = 1;
    }
    GateProducer *gate = new GateProducer(offset + start + (uint64_t) finished, rangeSize - finished, gateThreads);
    if (gateUsesAvx512()) {
        printf("Using %d threads for the CPU gate (with AVX-512)\n", gateThreads);
    } else {
        printf("Using %d threads for the CPU gate. This CPU doesn't have AVX-512, so the gate is a lot slower\n", gateThreads);
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
    while (true) {
        if (stopRequested && !stopping) {
            stopping = true;
            printf("Stopping, the batches on the GPU and the hit checks finish first...\n");
            fflush(stdout);
        }

        // Stop when all the indexes are queued (or we're stopping) and the batches are finished
        bool moreBatches = queued < rangeSize && !stopping;
        if (!moreBatches && !anyBatchInFlight(batches)) {
            break;
        }

        BatchSlot &batch = batches[slot];
        if (batch.inFlight) {
            finished += completeBatch(batch, hostResults, hostSeeds, results, finished);
            saveCheckedHits(results, settings, false, hitsFile);
        }

        // Start the next batch in this slot
        long span = 0;
        if (moreBatches && gate->nextBatch(hostIndexes, span)) {
            batch.survivors = runBatch(batch, hostIndexes, lut, stageTotals);
            batch.span = span;
            batch.inFlight = true;
            queued += span;
            //printf("slot %d: %u seeds went to the cascade\n", slot, batch.survivors);
        }
        slot = (slot + 1) % STREAM_COUNT;

        // Every 30 seconds save the checkpoint (and the speed)
        float elapsedMs = millisecondsSince(startEvent, nowEvent);
        if (elapsedMs / 1000.0 - lastSave >= 30.0) {
            lastSave = elapsedMs / 1000.0;
            saveSpeed(elapsedMs, finished - finishedAtStart);
            if (!saveCheckpoint(CHECKPOINT_PATH, offset, start, rangeSize, checkpointPosition(results, finished))) {
                return 1;
            }
        }
    }

    float runMs = millisecondsSince(startEvent, nowEvent);
    printf("The CPU gate sent %ld of %ld indexes to the GPU (%.3f%%)\n", (long) gate->sent, finished - finishedAtStart,
           100.0 * gate->sent / (double) std::max(1L, finished - finishedAtStart));
    delete gate;

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
