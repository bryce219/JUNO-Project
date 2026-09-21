// CPU gate for the scanner. This checks the y offsets of a few climate octaves before a seed goes to the GPU.
// Good seeds mostly have humidity, erosion and weirdness octaves with a y offset fraction close to a half, so we add up how far these
// are from a half (times a weight) and keep the seeds with a small total.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "rng.h"
#include "hostgate.h"

// The salts of the climate values we check
static const uint64_t CLIMATE_SALTS[3][2] = {
    {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL},  // Humidity
    {0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL},  // Erosion
    {0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL}}; // Weirdness

// md5 salts for the first two octaves of each one
static const uint64_t OCTAVE_SALTS[3][2][2] = {
    {{0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL},   // Humidity
     {0xf11268128982754fULL, 0x257a1d670430b0aaULL}},
    {{0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL},   // Erosion
     {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}},
    {{0xf11268128982754fULL, 0x257a1d670430b0aaULL},   // Weirdness
     {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL}}};

// Octave weights, [climate][half][octave]. I got these from a logistic regression
static const double GATE_WEIGHTS[3][2][2] = {
    {{1.000, 0.281}, {0.984, 0.202}},  // Humidity
    {{0.464, 0.245}, {0.486, 0.193}},  // Erosion
    {{0.197, 0.182}, {0.209, 0.198}}}; // Weirdness

// Copy of streamSeed from scan.cu
static uint64_t streamSeed(uint64_t index) {
    uint64_t mixed = index * 0x9E3779B97F4A7C15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    return mixed ^ (mixed >> 31);
}

/**
 * @brief Returns the gate total for a seed. The random numbers come out the same way they do when cubiomes sets up the climate noise.
 * 
 * @param seed 
 * @return double The total (the gate keeps the seed if it's less than or equal to the threshold)
 */
static double gateTotal(uint64_t seed) {
    Xoroshiro seedRandom;
    xSetSeed(&seedRandom, seed);
    uint64_t low = xNextLong(&seedRandom);
    uint64_t high = xNextLong(&seedRandom);

    double total = 0;
    for (int climate = 0; climate < 3; climate++) {
        Xoroshiro climateRandom = {low ^ CLIMATE_SALTS[climate][0], high ^ CLIMATE_SALTS[climate][1]};
        for (int half = 0; half < 2; half++) {
            uint64_t halfLow = xNextLong(&climateRandom);
            uint64_t halfHigh = xNextLong(&climateRandom);
            for (int octave = 0; octave < 2; octave++) {
                Xoroshiro octaveRandom = {halfLow ^ OCTAVE_SALTS[climate][octave][0], halfHigh ^ OCTAVE_SALTS[climate][octave][1]};
                xNextLong(&octaveRandom); // Skip the x offset
                double yOffset = xNextDouble(&octaveRandom) * 256.0;
                total += GATE_WEIGHTS[climate][half][octave] * fabs(yOffset - floor(yOffset) - 0.5);
            }
        }
    }
    return total;
}

/**
 * @brief This method runs the gate on one index at a time. We use it for the last few indexes, or when the CPU doesn't have
 * AVX-512.
 * 
 * @param threshold The gate threshold
 * @param firstIndex The stream index to start from
 * @param indexCount Indexes to check
 * @param output Gets the indexes that pass, in order
 * @return size_t Number of indexes that passed
 */
static size_t gateOneByOne(double threshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    size_t passCount = 0;
    for (size_t i = 0; i < indexCount; i++) {
        if (gateTotal(streamSeed(firstIndex + i)) <= threshold) {
            output[passCount++] = firstIndex + i;
        }
    }
    return passCount;
}

// GATE_WEIGHTS as floats, the way we have them on the GPU
static const float FLOAT_GATE_WEIGHTS[3][2][2] = {
    {{1.000f, 0.281f}, {0.984f, 0.202f}},  // Humidity
    {{0.464f, 0.245f}, {0.486f, 0.193f}},  // Erosion
    {{0.197f, 0.182f}, {0.209f, 0.198f}}}; // Weirdness

/**
 * @brief Returns the gate total for one climate value in floats, like gateClimate in gpu_gate.cuh
 * 
 * @param climate 0, 1 or 2 (humidity, erosion or weirdness)
 * @param low The low random number of the seed
 * @param high The high one
 * @return float The total for this climate value
 */
static float floatClimateTotal(int climate, uint64_t low, uint64_t high) {
    Xoroshiro climateRandom = {low ^ CLIMATE_SALTS[climate][0], high ^ CLIMATE_SALTS[climate][1]};
    float total = 0.0f;
    for (int half = 0; half < 2; half++) {
        uint64_t halfLow = xNextLong(&climateRandom);
        uint64_t halfHigh = xNextLong(&climateRandom);
        for (int octave = 0; octave < 2; octave++) {
            Xoroshiro octaveRandom = {halfLow ^ OCTAVE_SALTS[climate][octave][0], halfHigh ^ OCTAVE_SALTS[climate][octave][1]};
            xNextLong(&octaveRandom); // Skip the x offset
            int fraction = (int) ((uint32_t) (xNextLong(&octaveRandom) >> 32) & 0xFFFFFFu);
            total = fmaf(FLOAT_GATE_WEIGHTS[climate][half][octave], (float) abs(fraction - 0x800000), total); // how far it is from a half, times 2^24
        }
    }
    return total;
}

// true if the GPU gate keeps the seed, checked in gateKernel's order
static int passesFloatGate(uint64_t seed, float threshold, float humidityThreshold) {
    Xoroshiro seedRandom;
    xSetSeed(&seedRandom, seed);
    uint64_t low = xNextLong(&seedRandom);
    uint64_t high = xNextLong(&seedRandom);

    float total = floatClimateTotal(0, low, high);
    if (!(total <= humidityThreshold)) {
        return 0;
    }
    total += floatClimateTotal(1, low, high);
    if (!(total <= threshold)) {
        return 0;
    }
    total += floatClimateTotal(2, low, high);
    return total <= threshold;
}

// gateOneByOne for the float gate
static size_t floatGateOneByOne(float threshold, float humidityThreshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    size_t passCount = 0;
    for (size_t i = 0; i < indexCount; i++) {
        if (passesFloatGate(streamSeed(firstIndex + i), threshold, humidityThreshold)) {
            output[passCount++] = firstIndex + i;
        }
    }
    return passCount;
}

// AVX-512 version of the gate, it does 8 seeds at a time
#ifdef __AVX512DQ__
#include <immintrin.h>

// 8 random generators side by side
typedef struct {
    __m512i low;
    __m512i high;
} RandomVectors;

// xNextLong for all 8
static inline __m512i nextLongs(RandomVectors *random) {
    __m512i low = random->low;
    __m512i high = random->high;
    __m512i result = _mm512_add_epi64(_mm512_rol_epi64(_mm512_add_epi64(low, high), 17), low);

    high = _mm512_xor_si512(high, low);
    random->low = _mm512_xor_si512(_mm512_xor_si512(_mm512_rol_epi64(low, 49), high), _mm512_slli_epi64(high, 21));
    random->high = _mm512_rol_epi64(high, 28);
    return result;
}

// Mixes the bits like streamSeed and xSetSeed do
static inline __m512i mixBits(__m512i value, uint64_t firstMultiplier, uint64_t secondMultiplier) {
    value = _mm512_mullo_epi64(_mm512_xor_si512(value, _mm512_srli_epi64(value, 30)), _mm512_set1_epi64((long long) firstMultiplier));
    value = _mm512_mullo_epi64(_mm512_xor_si512(value, _mm512_srli_epi64(value, 27)), _mm512_set1_epi64((long long) secondMultiplier));
    return _mm512_xor_si512(value, _mm512_srli_epi64(value, 31));
}

// How far the y offset fraction is from a half. This only looks at the top bits of the fraction
static inline __m512d distanceFromHalf(__m512i randomLongs) {
    __m512i fractionBits = _mm512_and_si512(_mm512_srli_epi64(randomLongs, 32), _mm512_set1_epi64(0xFFFFFF));
    __m512d distance = _mm512_sub_pd(_mm512_mul_pd(_mm512_cvtepi64_pd(fractionBits), _mm512_set1_pd(0x1.0p-24)),
                                     _mm512_set1_pd(0.5));
    return _mm512_abs_pd(distance);
}

/**
 * @brief Gets the random numbers the seeds start with, for 8 stream indexes
 * @param indexes The stream indexes
 * @param low Gets set to the low random numbers of the seeds
 * @param high The high ones
 */
static inline void seedRandoms(__m512i indexes, __m512i *low, __m512i *high) {
    // stream seeds
    __m512i mixed = _mm512_mullo_epi64(indexes, _mm512_set1_epi64((long long) 0x9E3779B97F4A7C15ULL));
    __m512i seeds = mixBits(mixed, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL);

    // then seed the generators
    __m512i stateLow = _mm512_xor_si512(seeds, _mm512_set1_epi64((long long) 0x6a09e667f3bcc909ULL));
    __m512i stateHigh = _mm512_add_epi64(stateLow, _mm512_set1_epi64((long long) 0x9e3779b97f4a7c15ULL));
    RandomVectors random = {mixBits(stateLow, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL),
                            mixBits(stateHigh, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL)};

    *low = nextLongs(&random);
    *high = nextLongs(&random);
}

/**
 * @brief Adds the octaves of a climate value to the gate totals, like gateTotal does
 * @param totals The totals so far
 * @param climate 0, 1 or 2 (humidity, erosion or weirdness)
 * @param low The low random numbers of the seeds
 * @param high The high random numbers of the seeds
 * @return __m512d The new totals
 */
static inline __m512d addClimate(__m512d totals, int climate, __m512i low, __m512i high, const __m512d weights[3][2][2]) {
    RandomVectors climateRandom = {_mm512_xor_si512(low, _mm512_set1_epi64((long long) CLIMATE_SALTS[climate][0])),
                                   _mm512_xor_si512(high, _mm512_set1_epi64((long long) CLIMATE_SALTS[climate][1]))};
    for (int half = 0; half < 2; half++) {
        __m512i halfLow = nextLongs(&climateRandom);
        __m512i halfHigh = nextLongs(&climateRandom);
        for (int octave = 0; octave < 2; octave++) {
            RandomVectors octaveRandom = {_mm512_xor_si512(halfLow, _mm512_set1_epi64((long long) OCTAVE_SALTS[climate][octave][0])),
                                          _mm512_xor_si512(halfHigh, _mm512_set1_epi64((long long) OCTAVE_SALTS[climate][octave][1]))};
            nextLongs(&octaveRandom); //skip the x offset
            totals = _mm512_add_pd(totals, _mm512_mul_pd(weights[climate][half][octave], distanceFromHalf(nextLongs(&octaveRandom))));
        }
    }
    return totals;
}

// Copies the weights into AVX-512 vectors
static void loadWeights(__m512d weights[3][2][2]) {
    for (int climate = 0; climate < 3; climate++) {
        for (int half = 0; half < 2; half++) {
            for (int octave = 0; octave < 2; octave++) {
                weights[climate][half][octave] = _mm512_set1_pd(GATE_WEIGHTS[climate][half][octave]);
            }
        }
    }
}

#define QUEUE_SIZE 8192     // indexes that go through the gate together
#define HUMIDITY_LIMIT 0.50 // Biggest humidity total we keep. This loses around 1 passing seed in 20,000 (gateOneByOne doesn't have the limit)

/**
 * @brief This method runs the gate on the indexes 8 at a time. Humidity gets added up for all of them, then erosion and weirdness only for the seeds that are left.
 * 
 * @param threshold The gate threshold
 * @param firstIndex The stream index to start from
 * @param indexCount The number of indexes to check
 * @param output Where the passing indexes go, in order
 * @return size_t The number of indexes that passed
 */
size_t gateIndexes(double threshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    const __m512i indexOffsets = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512d thresholds = _mm512_set1_pd(threshold);
    const __m512d humidityThresholds = _mm512_set1_pd(threshold < HUMIDITY_LIMIT ? threshold : HUMIDITY_LIMIT);
    __m512d weights[3][2][2];
    loadWeights(weights);

    // Seeds left after each climate value: the index, random numbers and total so far
    static __thread uint64_t queueIndexes[2][QUEUE_SIZE + 8];
    static __thread uint64_t queueLows[2][QUEUE_SIZE + 8];
    static __thread uint64_t queueHighs[2][QUEUE_SIZE + 8];
    static __thread double queueTotals[2][QUEUE_SIZE + 8];

    size_t passCount = 0;
    size_t i = 0;
    while (i < indexCount) {
        size_t queued = 0;
        size_t end = i + QUEUE_SIZE;
        if (end > indexCount) {
            end = indexCount;
        }

        // Humidity
        for (; i + 8 <= end; i += 8) {
            __m512i indexes = _mm512_add_epi64(_mm512_set1_epi64((long long) (firstIndex + i)), indexOffsets);
            __m512i low;
            __m512i high;
            seedRandoms(indexes, &low, &high);

            __m512d totals = addClimate(_mm512_setzero_pd(), 0, low, high, weights);
            __mmask8 passed = _mm512_cmp_pd_mask(totals, humidityThresholds, _CMP_LE_OQ);
            if (passed) {
                _mm512_storeu_si512(queueIndexes[0] + queued, _mm512_maskz_compress_epi64(passed, indexes));
                _mm512_storeu_si512(queueLows[0] + queued, _mm512_maskz_compress_epi64(passed, low));
                _mm512_storeu_si512(queueHighs[0] + queued, _mm512_maskz_compress_epi64(passed, high));
                _mm512_storeu_pd(queueTotals[0] + queued, _mm512_maskz_compress_pd(passed, totals));
                queued += (size_t) __builtin_popcount(passed);
            }
        }

        // The last few indexes of the range go through one by one
        uint64_t leftoverIndexes[8];
        size_t leftoverCount = gateOneByOne(threshold, firstIndex + i, end - i, leftoverIndexes);
        i = end;

        // Erosion, then weirdness
        for (int climate = 1; climate <= 2; climate++) {
            int source = climate == 1 ? 0 : 1;
            int destination = climate == 1 ? 1 : 0;

            // Fill the rest of the the last group of 8 with seeds that can't pass
            for (size_t j = queued; j < ((queued + 7) & ~7UL); j++) {
                queueIndexes[source][j] = 0;
                queueLows[source][j] = 0;
                queueHighs[source][j] = 0;
                queueTotals[source][j] = 1e9;
            }

            size_t kept = 0;
            for (size_t j = 0; j < queued; j += 8) {
                __m512i indexes = _mm512_loadu_si512(queueIndexes[source] + j);
                __m512i low = _mm512_loadu_si512(queueLows[source] + j);
                __m512i high = _mm512_loadu_si512(queueHighs[source] + j);
                __m512d totals = addClimate(_mm512_loadu_pd(queueTotals[source] + j), climate, low, high, weights);
                __mmask8 passed = _mm512_cmp_pd_mask(totals, thresholds, _CMP_LE_OQ);
                if (passed) {
                    _mm512_storeu_si512(queueIndexes[destination] + kept, _mm512_maskz_compress_epi64(passed, indexes));
                    // weirdness still needs the random numbers and totals
                    if (climate == 1) {
                        _mm512_storeu_si512(queueLows[destination] + kept, _mm512_maskz_compress_epi64(passed, low));
                        _mm512_storeu_si512(queueHighs[destination] + kept, _mm512_maskz_compress_epi64(passed, high));
                        _mm512_storeu_pd(queueTotals[destination] + kept, _mm512_maskz_compress_pd(passed, totals));
                    }
                    kept += (size_t) __builtin_popcount(passed);
                }
            }
            queued = kept;
        }

        // The queue is in index order and the leftovers come after it, so the output stays in order
        if (queued) {
            memcpy(output + passCount, queueIndexes[0], queued * 8);
            passCount += queued;
        }
        for (size_t j = 0; j < leftoverCount; j++) {
            output[passCount++] = leftoverIndexes[j];
        }
    }
    return passCount;
}


#if defined(__AVX512VL__) && defined(__FMA__)
// floatClimateTotal for 8 seeds
static inline __m256 floatClimateTotals(int climate, __m512i low, __m512i high, const __m256 weights[3][2][2]) {
    RandomVectors climateRandom = {_mm512_xor_si512(low, _mm512_set1_epi64((long long) CLIMATE_SALTS[climate][0])),
                                   _mm512_xor_si512(high, _mm512_set1_epi64((long long) CLIMATE_SALTS[climate][1]))};
    __m256 totals = _mm256_setzero_ps();
    for (int half = 0; half < 2; half++) {
        __m512i halfLow = nextLongs(&climateRandom);
        __m512i halfHigh = nextLongs(&climateRandom);
        for (int octave = 0; octave < 2; octave++) {
            RandomVectors octaveRandom = {_mm512_xor_si512(halfLow, _mm512_set1_epi64((long long) OCTAVE_SALTS[climate][octave][0])),
                                          _mm512_xor_si512(halfHigh, _mm512_set1_epi64((long long) OCTAVE_SALTS[climate][octave][1]))};
            nextLongs(&octaveRandom); // skip the x offset
            __m512i fractionBits = _mm512_and_si512(_mm512_srli_epi64(nextLongs(&octaveRandom), 32), _mm512_set1_epi64(0xFFFFFF));
            __m256i distances = _mm256_abs_epi32(_mm256_sub_epi32(_mm512_cvtepi64_epi32(fractionBits), _mm256_set1_epi32(0x800000)));
            totals = _mm256_fmadd_ps(weights[climate][half][octave], _mm256_cvtepi32_ps(distances), totals);
        }
    }
    return totals;
}

/**
 * @brief This method runs the float gate 8 indexes at a time, like gateIndexes
 * 
 * @param threshold The GPU gate's threshold
 * @param humidityThreshold The GPU gate's humidity cut
 */
size_t floatGateIndexes(float threshold, float humidityThreshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    const __m512i indexOffsets = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256 thresholds = _mm256_set1_ps(threshold);
    const __m256 humidityThresholds = _mm256_set1_ps(humidityThreshold);
    __m256 weights[3][2][2];
    for (int climate = 0; climate < 3; climate++) {
        for (int half = 0; half < 2; half++) {
            for (int octave = 0; octave < 2; octave++) {
                weights[climate][half][octave] = _mm256_set1_ps(FLOAT_GATE_WEIGHTS[climate][half][octave]);
            }
        }
    }

    static __thread uint64_t queueIndexes[2][QUEUE_SIZE + 8];
    static __thread uint64_t queueLows[2][QUEUE_SIZE + 8];
    static __thread uint64_t queueHighs[2][QUEUE_SIZE + 8];
    static __thread float queueTotals[2][QUEUE_SIZE + 8];

    size_t passCount = 0;
    size_t i = 0;
    while (i < indexCount) {
        size_t queued = 0;
        size_t end = i + QUEUE_SIZE;
        if (end > indexCount) {
            end = indexCount;
        }

        // Humidity
        for (; i + 8 <= end; i += 8) {
            __m512i indexes = _mm512_add_epi64(_mm512_set1_epi64((long long) (firstIndex + i)), indexOffsets);
            __m512i low;
            __m512i high;
            seedRandoms(indexes, &low, &high);

            __m256 totals = floatClimateTotals(0, low, high, weights);
            __mmask8 passed = _mm256_cmp_ps_mask(totals, humidityThresholds, _CMP_LE_OQ);
            if (passed) {
                _mm512_storeu_si512(queueIndexes[0] + queued, _mm512_maskz_compress_epi64(passed, indexes));
                _mm512_storeu_si512(queueLows[0] + queued, _mm512_maskz_compress_epi64(passed, low));
                _mm512_storeu_si512(queueHighs[0] + queued, _mm512_maskz_compress_epi64(passed, high));
                _mm256_storeu_ps(queueTotals[0] + queued, _mm256_maskz_compress_ps(passed, totals));
                queued += (size_t) __builtin_popcount(passed);
            }
        }

        // The last few indexes of the range go through one by one
        uint64_t leftoverIndexes[8];
        size_t leftoverCount = floatGateOneByOne(threshold, humidityThreshold, firstIndex + i, end - i, leftoverIndexes);
        i = end;

        // Erosion, then weirdness
        for (int climate = 1; climate <= 2; climate++) {
            int source = climate == 1 ? 0 : 1;
            int destination = climate == 1 ? 1 : 0;

            // Fill the rest of the last group of 8 with seeds that can't pass
            for (size_t j = queued; j < ((queued + 7) & ~7UL); j++) {
                queueIndexes[source][j] = 0;
                queueLows[source][j] = 0;
                queueHighs[source][j] = 0;
                queueTotals[source][j] = 1e9f;
            }

            size_t kept = 0;
            for (size_t j = 0; j < queued; j += 8) {
                __m512i indexes = _mm512_loadu_si512(queueIndexes[source] + j);
                __m512i low = _mm512_loadu_si512(queueLows[source] + j);
                __m512i high = _mm512_loadu_si512(queueHighs[source] + j);
                __m256 totals = _mm256_add_ps(_mm256_loadu_ps(queueTotals[source] + j), floatClimateTotals(climate, low, high, weights));
                __mmask8 passed = _mm256_cmp_ps_mask(totals, thresholds, _CMP_LE_OQ);
                if (passed) {
                    _mm512_storeu_si512(queueIndexes[destination] + kept, _mm512_maskz_compress_epi64(passed, indexes));
                    if (climate == 1) {
                        _mm512_storeu_si512(queueLows[destination] + kept, _mm512_maskz_compress_epi64(passed, low));
                        _mm512_storeu_si512(queueHighs[destination] + kept, _mm512_maskz_compress_epi64(passed, high));
                        _mm256_storeu_ps(queueTotals[destination] + kept, _mm256_maskz_compress_ps(passed, totals));
                    }
                    kept += (size_t) __builtin_popcount(passed);
                }
            }
            queued = kept;
        }

        if (queued) {
            memcpy(output + passCount, queueIndexes[0], queued * 8);
            passCount += queued;
        }
        for (size_t j = 0; j < leftoverCount; j++) {
            output[passCount++] = leftoverIndexes[j];
        }
    }
    return passCount;
}
#else
size_t floatGateIndexes(float threshold, float humidityThreshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    return floatGateOneByOne(threshold, humidityThreshold, firstIndex, indexCount, output);
}
#endif


int gateLanes(void) {
    return 8;
}

// AVX2 version of the gate, it does 4 seeds at a time
#elif defined(__AVX2__)
#include <immintrin.h>

// 4 random generators side by side
typedef struct {
    __m256i low;
    __m256i high;
} RandomVectors;

// AVX2 has no rotate, so it takes a pair of shifts
static inline __m256i rotateLeft(__m256i value, int bits) {
    return _mm256_or_si256(_mm256_slli_epi64(value, bits), _mm256_srli_epi64(value, 64 - bits));
}

// No whole multiply either, so this builds one out of the halves
static inline __m256i multiplyLow(__m256i a, __m256i b) {
    __m256i aHigh = _mm256_srli_epi64(a, 32);
    __m256i bHigh = _mm256_srli_epi64(b, 32);
    __m256i lowProduct = _mm256_mul_epu32(a, b);
    __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(a, bHigh), _mm256_mul_epu32(aHigh, b));
    return _mm256_add_epi64(lowProduct, _mm256_slli_epi64(cross, 32));
}

// xNextLong for all 4
static inline __m256i nextLongs(RandomVectors *random) {
    __m256i low = random->low;
    __m256i high = random->high;
    __m256i result = _mm256_add_epi64(rotateLeft(_mm256_add_epi64(low, high), 17), low);

    high = _mm256_xor_si256(high, low);
    random->low = _mm256_xor_si256(_mm256_xor_si256(rotateLeft(low, 49), high), _mm256_slli_epi64(high, 21));
    random->high = rotateLeft(high, 28);
    return result;
}

// Mixes the bits like streamSeed and xSetSeed do
static inline __m256i mixBits(__m256i value, uint64_t firstMultiplier, uint64_t secondMultiplier) {
    value = multiplyLow(_mm256_xor_si256(value, _mm256_srli_epi64(value, 30)), _mm256_set1_epi64x((long long) firstMultiplier));
    value = multiplyLow(_mm256_xor_si256(value, _mm256_srli_epi64(value, 27)), _mm256_set1_epi64x((long long) secondMultiplier));
    return _mm256_xor_si256(value, _mm256_srli_epi64(value, 31));
}

// How far the y offset fraction is from a half. The top bits go to a double through an int
static inline __m256d distanceFromHalf(__m256i randomLongs) {
    __m256i fractionBits = _mm256_and_si256(_mm256_srli_epi64(randomLongs, 32), _mm256_set1_epi64x(0xFFFFFF));
    __m128i packed = _mm256_castsi256_si128(
        _mm256_permutevar8x32_epi32(fractionBits, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)));
    __m256d distance = _mm256_sub_pd(_mm256_mul_pd(_mm256_cvtepi32_pd(packed), _mm256_set1_pd(0x1.0p-24)),
                                     _mm256_set1_pd(0.5));
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), distance);
}

// No compress either, so a permute stands in for it
static void loadCompressMoves(__m256i moves[16]) {
    for (int passed = 0; passed < 16; passed++) {
        int32_t lanes[8] = {0};
        int slot = 0;
        for (int lane = 0; lane < 4; lane++) {
            if (passed & (1 << lane)) {
                lanes[slot++] = lane + lane;
                lanes[slot++] = lane + lane + 1;
            }
        }
        moves[passed] = _mm256_loadu_si256((const __m256i *) lanes);
    }
}

static inline __m256i compressLongs(__m256i value, __m256i move) {
    return _mm256_permutevar8x32_epi32(value, move);
}

static inline __m256d compressDoubles(__m256d value, __m256i move) {
    return _mm256_castsi256_pd(compressLongs(_mm256_castpd_si256(value), move));
}

// Same as the AVX-512 seedRandoms, 4 at a time
static inline void seedRandoms(__m256i indexes, __m256i *low, __m256i *high) {
    __m256i mixed = multiplyLow(indexes, _mm256_set1_epi64x((long long) 0x9E3779B97F4A7C15ULL));
    __m256i seeds = mixBits(mixed, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL);

    __m256i stateLow = _mm256_xor_si256(seeds, _mm256_set1_epi64x((long long) 0x6a09e667f3bcc909ULL));
    __m256i stateHigh = _mm256_add_epi64(stateLow, _mm256_set1_epi64x((long long) 0x9e3779b97f4a7c15ULL));
    RandomVectors random = {mixBits(stateLow, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL),
                            mixBits(stateHigh, 0xbf58476d1ce4e5b9ULL, 0x94d049bb133111ebULL)};

    *low = nextLongs(&random);
    *high = nextLongs(&random);
}

// Same as the AVX-512 addClimate, 4 at a time
static inline __m256d addClimate(__m256d totals, int climate, __m256i low, __m256i high, const __m256d weights[3][2][2]) {
    RandomVectors climateRandom = {_mm256_xor_si256(low, _mm256_set1_epi64x((long long) CLIMATE_SALTS[climate][0])),
                                   _mm256_xor_si256(high, _mm256_set1_epi64x((long long) CLIMATE_SALTS[climate][1]))};
    for (int half = 0; half < 2; half++) {
        __m256i halfLow = nextLongs(&climateRandom);
        __m256i halfHigh = nextLongs(&climateRandom);
        for (int octave = 0; octave < 2; octave++) {
            RandomVectors octaveRandom = {_mm256_xor_si256(halfLow, _mm256_set1_epi64x((long long) OCTAVE_SALTS[climate][octave][0])),
                                          _mm256_xor_si256(halfHigh, _mm256_set1_epi64x((long long) OCTAVE_SALTS[climate][octave][1]))};
            nextLongs(&octaveRandom); // skip the x offset
            totals = _mm256_add_pd(totals, _mm256_mul_pd(weights[climate][half][octave], distanceFromHalf(nextLongs(&octaveRandom))));
        }
    }
    return totals;
}

// Copies the weights into AVX2 vectors
static void loadWeights(__m256d weights[3][2][2]) {
    for (int climate = 0; climate < 3; climate++) {
        for (int half = 0; half < 2; half++) {
            for (int octave = 0; octave < 2; octave++) {
                weights[climate][half][octave] = _mm256_set1_pd(GATE_WEIGHTS[climate][half][octave]);
            }
        }
    }
}

#define QUEUE_SIZE 8192     // indexes that go through the gate together
#define HUMIDITY_LIMIT 0.50 // The same cut the AVX-512 path takes, so the two agree

// Runs the gate 4 indexes at a time, the same way the AVX-512 one runs 8
size_t gateIndexes(double threshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    const __m256i indexOffsets = _mm256_set_epi64x(3, 2, 1, 0);
    const __m256d thresholds = _mm256_set1_pd(threshold);
    const __m256d humidityThresholds = _mm256_set1_pd(threshold < HUMIDITY_LIMIT ? threshold : HUMIDITY_LIMIT);
    __m256d weights[3][2][2];
    __m256i moves[16];
    loadWeights(weights);
    loadCompressMoves(moves);

    static __thread uint64_t queueIndexes[2][QUEUE_SIZE + 4];
    static __thread uint64_t queueLows[2][QUEUE_SIZE + 4];
    static __thread uint64_t queueHighs[2][QUEUE_SIZE + 4];
    static __thread double queueTotals[2][QUEUE_SIZE + 4];

    size_t passCount = 0;
    size_t i = 0;
    while (i < indexCount) {
        size_t queued = 0;
        size_t end = i + QUEUE_SIZE;
        if (end > indexCount) {
            end = indexCount;
        }

        // Humidity
        for (; i + 4 <= end; i += 4) {
            __m256i indexes = _mm256_add_epi64(_mm256_set1_epi64x((long long) (firstIndex + i)), indexOffsets);
            __m256i low;
            __m256i high;
            seedRandoms(indexes, &low, &high);

            __m256d totals = addClimate(_mm256_setzero_pd(), 0, low, high, weights);
            int passed = _mm256_movemask_pd(_mm256_cmp_pd(totals, humidityThresholds, _CMP_LE_OQ));
            if (passed) {
                _mm256_storeu_si256((__m256i *) (queueIndexes[0] + queued), compressLongs(indexes, moves[passed]));
                _mm256_storeu_si256((__m256i *) (queueLows[0] + queued), compressLongs(low, moves[passed]));
                _mm256_storeu_si256((__m256i *) (queueHighs[0] + queued), compressLongs(high, moves[passed]));
                _mm256_storeu_pd(queueTotals[0] + queued, compressDoubles(totals, moves[passed]));
                queued += (size_t) __builtin_popcount((unsigned) passed);
            }
        }

        // The last few indexes of the range go through one by one
        uint64_t leftoverIndexes[4];
        size_t leftoverCount = gateOneByOne(threshold, firstIndex + i, end - i, leftoverIndexes);
        i = end;

        // Erosion, then weirdness
        for (int climate = 1; climate <= 2; climate++) {
            int source = climate == 1 ? 0 : 1;
            int destination = climate == 1 ? 1 : 0;

            // Fill the rest of the last group of 4 with seeds that can't pass
            for (size_t j = queued; j < ((queued + 3) & ~3UL); j++) {
                queueIndexes[source][j] = 0;
                queueLows[source][j] = 0;
                queueHighs[source][j] = 0;
                queueTotals[source][j] = 1e9;
            }

            size_t kept = 0;
            for (size_t j = 0; j < queued; j += 4) {
                __m256i indexes = _mm256_loadu_si256((const __m256i *) (queueIndexes[source] + j));
                __m256i low = _mm256_loadu_si256((const __m256i *) (queueLows[source] + j));
                __m256i high = _mm256_loadu_si256((const __m256i *) (queueHighs[source] + j));
                __m256d totals = addClimate(_mm256_loadu_pd(queueTotals[source] + j), climate, low, high, weights);
                int passed = _mm256_movemask_pd(_mm256_cmp_pd(totals, thresholds, _CMP_LE_OQ));
                if (passed) {
                    _mm256_storeu_si256((__m256i *) (queueIndexes[destination] + kept), compressLongs(indexes, moves[passed]));
                    // weirdness still needs the random numbers and totals
                    if (climate == 1) {
                        _mm256_storeu_si256((__m256i *) (queueLows[destination] + kept), compressLongs(low, moves[passed]));
                        _mm256_storeu_si256((__m256i *) (queueHighs[destination] + kept), compressLongs(high, moves[passed]));
                        _mm256_storeu_pd(queueTotals[destination] + kept, compressDoubles(totals, moves[passed]));
                    }
                    kept += (size_t) __builtin_popcount((unsigned) passed);
                }
            }
            queued = kept;
        }

        // The queue is in index order and the leftovers come after it, so the output stays in order
        if (queued) {
            memcpy(output + passCount, queueIndexes[0], queued * 8);
            passCount += queued;
        }
        for (size_t j = 0; j < leftoverCount; j++) {
            output[passCount++] = leftoverIndexes[j];
        }
    }
    return passCount;
}


#ifdef __FMA__
// floatClimateTotal for 4 seeds
static inline __m128 floatClimateTotals(int climate, __m256i low, __m256i high, const __m128 weights[3][2][2]) {
    RandomVectors climateRandom = {_mm256_xor_si256(low, _mm256_set1_epi64x((long long) CLIMATE_SALTS[climate][0])),
                                   _mm256_xor_si256(high, _mm256_set1_epi64x((long long) CLIMATE_SALTS[climate][1]))};
    __m128 totals = _mm_setzero_ps();
    for (int half = 0; half < 2; half++) {
        __m256i halfLow = nextLongs(&climateRandom);
        __m256i halfHigh = nextLongs(&climateRandom);
        for (int octave = 0; octave < 2; octave++) {
            RandomVectors octaveRandom = {_mm256_xor_si256(halfLow, _mm256_set1_epi64x((long long) OCTAVE_SALTS[climate][octave][0])),
                                          _mm256_xor_si256(halfHigh, _mm256_set1_epi64x((long long) OCTAVE_SALTS[climate][octave][1]))};
            nextLongs(&octaveRandom); // skip the x offset
            __m256i fractionBits = _mm256_and_si256(_mm256_srli_epi64(nextLongs(&octaveRandom), 32), _mm256_set1_epi64x(0xFFFFFF));
            __m128i fractions = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(fractionBits, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)));
            __m128i distances = _mm_abs_epi32(_mm_sub_epi32(fractions, _mm_set1_epi32(0x800000)));
            totals = _mm_fmadd_ps(weights[climate][half][octave], _mm_cvtepi32_ps(distances), totals);
        }
    }
    return totals;
}

// Runs the float gate 4 indexes at a time, like the AVX-512 one runs 8
size_t floatGateIndexes(float threshold, float humidityThreshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    const __m256i indexOffsets = _mm256_set_epi64x(3, 2, 1, 0);
    const __m128 thresholds = _mm_set1_ps(threshold);
    const __m128 humidityThresholds = _mm_set1_ps(humidityThreshold);
    __m128 weights[3][2][2];
    for (int climate = 0; climate < 3; climate++) {
        for (int half = 0; half < 2; half++) {
            for (int octave = 0; octave < 2; octave++) {
                weights[climate][half][octave] = _mm_set1_ps(FLOAT_GATE_WEIGHTS[climate][half][octave]);
            }
        }
    }
    __m256i moves[16];
    __m128i floatMoves[16]; // moves for 4 floats
    loadCompressMoves(moves);
    for (int passed = 0; passed < 16; passed++) {
        int32_t lanes[4] = {0};
        int slot = 0;
        for (int lane = 0; lane < 4; lane++) {
            if (passed & (1 << lane)) {
                lanes[slot++] = lane;
            }
        }
        floatMoves[passed] = _mm_loadu_si128((const __m128i *) lanes);
    }

    static __thread uint64_t queueIndexes[2][QUEUE_SIZE + 4];
    static __thread uint64_t queueLows[2][QUEUE_SIZE + 4];
    static __thread uint64_t queueHighs[2][QUEUE_SIZE + 4];
    static __thread float queueTotals[2][QUEUE_SIZE + 4];

    size_t passCount = 0;
    size_t i = 0;
    while (i < indexCount) {
        size_t queued = 0;
        size_t end = i + QUEUE_SIZE;
        if (end > indexCount) {
            end = indexCount;
        }

        // Humidity
        for (; i + 4 <= end; i += 4) {
            __m256i indexes = _mm256_add_epi64(_mm256_set1_epi64x((long long) (firstIndex + i)), indexOffsets);
            __m256i low;
            __m256i high;
            seedRandoms(indexes, &low, &high);

            __m128 totals = floatClimateTotals(0, low, high, weights);
            int passed = _mm_movemask_ps(_mm_cmp_ps(totals, humidityThresholds, _CMP_LE_OQ));
            if (passed) {
                _mm256_storeu_si256((__m256i *) (queueIndexes[0] + queued), compressLongs(indexes, moves[passed]));
                _mm256_storeu_si256((__m256i *) (queueLows[0] + queued), compressLongs(low, moves[passed]));
                _mm256_storeu_si256((__m256i *) (queueHighs[0] + queued), compressLongs(high, moves[passed]));
                _mm_storeu_ps(queueTotals[0] + queued, _mm_permutevar_ps(totals, floatMoves[passed]));
                queued += (size_t) __builtin_popcount((unsigned) passed);
            }
        }

        // The last few indexes of the range go through one by one
        uint64_t leftoverIndexes[4];
        size_t leftoverCount = floatGateOneByOne(threshold, humidityThreshold, firstIndex + i, end - i, leftoverIndexes);
        i = end;

        // Erosion, then weirdness
        for (int climate = 1; climate <= 2; climate++) {
            int source = climate == 1 ? 0 : 1;
            int destination = climate == 1 ? 1 : 0;

            // Fill the rest of the last group of 4 with seeds that can't pass
            for (size_t j = queued; j < ((queued + 3) & ~3UL); j++) {
                queueIndexes[source][j] = 0;
                queueLows[source][j] = 0;
                queueHighs[source][j] = 0;
                queueTotals[source][j] = 1e9f;
            }

            size_t kept = 0;
            for (size_t j = 0; j < queued; j += 4) {
                __m256i indexes = _mm256_loadu_si256((const __m256i *) (queueIndexes[source] + j));
                __m256i low = _mm256_loadu_si256((const __m256i *) (queueLows[source] + j));
                __m256i high = _mm256_loadu_si256((const __m256i *) (queueHighs[source] + j));
                __m128 totals = _mm_add_ps(_mm_loadu_ps(queueTotals[source] + j), floatClimateTotals(climate, low, high, weights));
                int passed = _mm_movemask_ps(_mm_cmp_ps(totals, thresholds, _CMP_LE_OQ));
                if (passed) {
                    _mm256_storeu_si256((__m256i *) (queueIndexes[destination] + kept), compressLongs(indexes, moves[passed]));
                    if (climate == 1) {
                        _mm256_storeu_si256((__m256i *) (queueLows[destination] + kept), compressLongs(low, moves[passed]));
                        _mm256_storeu_si256((__m256i *) (queueHighs[destination] + kept), compressLongs(high, moves[passed]));
                        _mm_storeu_ps(queueTotals[destination] + kept, _mm_permutevar_ps(totals, floatMoves[passed]));
                    }
                    kept += (size_t) __builtin_popcount((unsigned) passed);
                }
            }
            queued = kept;
        }

        if (queued) {
            memcpy(output + passCount, queueIndexes[0], queued * 8);
            passCount += queued;
        }
        for (size_t j = 0; j < leftoverCount; j++) {
            output[passCount++] = leftoverIndexes[j];
        }
    }
    return passCount;
}
#else
size_t floatGateIndexes(float threshold, float humidityThreshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    return floatGateOneByOne(threshold, humidityThreshold, firstIndex, indexCount, output);
}
#endif


int gateLanes(void) {
    return 4;
}
#else
// Without AVX-512 or AVX2 the gate checks the indexes one by one
size_t gateIndexes(double threshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    return gateOneByOne(threshold, firstIndex, indexCount, output);
}

size_t floatGateIndexes(float threshold, float humidityThreshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    return floatGateOneByOne(threshold, humidityThreshold, firstIndex, indexCount, output);
}


int gateLanes(void) {
    return 1;
}
#endif
