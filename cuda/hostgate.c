// CPU gate for the scanner. This checks the y offsets of a few climate octaves before a seed goes to the GPU.
// Good seeds mostly have humidity, erosion and weirdness octaves with a y offset fraction close to a half, so we add up how far these
// are from a half (times a weight) and keep the seeds with a small total.
#include <stdint.h>
#include <stddef.h>
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

int gateUsesAvx512(void) {
    return 1;
}
#else
// Without AVX-512 the gate checks the indexes one by one, which is about 5 times slower on a thread.
// TODO maybe an AVX2 version for these CPUs
size_t gateIndexes(double threshold, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    return gateOneByOne(threshold, firstIndex, indexCount, output);
}

int gateUsesAvx512(void) {
    return 0;
}
#endif
