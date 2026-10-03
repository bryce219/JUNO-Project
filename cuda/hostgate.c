// The CPU version of the gate (for --cpu-gate or --cpu-assist). It keeps the seeds gateKernel in gpu_gate.cuh keeps.
// Good seeds mostly have humidity, erosion and weirdness octaves with a y offset fraction close to a half. The gate adds up
// the distance from a half (times a weight). Seeds with a small total pass.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "hostgate.h"

static inline uint64_t rotateLeft(uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

// Returns the number xoroshiro would give for this state (xNextLong in cubiomes does this and then a step)
static inline uint64_t gateOutput(uint64_t low, uint64_t high) {
    return rotateLeft(low + high, 17) + low;
}

// A xoroshiro step, leaving out the number
static inline void gateStep(uint64_t *low, uint64_t *high) {
    uint64_t mixed = *high ^ *low;
    *low = rotateLeft(*low, 49) ^ mixed ^ (mixed << 21);
    *high = rotateLeft(mixed, 28);
}

// Mixes the bits of a seed like xSetSeed does
static inline uint64_t mixBits(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

/**
 * @brief Returns the distance between an octave's y offset fraction and a half (GATE_SCALE has the units)
 *
 * @param low The low half of the octave's generator
 * @param high The high half
 * @return uint32_t The distance from a half
 */
static inline uint32_t gateTerm(uint64_t low, uint64_t high) {
    // The high half of the output, leaving out the carry from the low half (like gateTerm in gpu_gate.cuh)
    uint32_t outputHigh = (uint32_t) (rotateLeft(low + high, 17) >> 32) + (uint32_t) (low >> 32);
    int32_t centered = (int32_t) (outputHigh & 0xFFFFFFu) - 0x800000;
    return (uint32_t) (centered < 0 ? -centered : centered) >> 7;
}

// The stepped octave salts (stepSalts fills this in when the program starts)
static uint64_t steppedSalts[3][2][2];

__attribute__((constructor)) static void stepSalts(void) {
    for (int climate = 0; climate < 3; climate++) {
        for (int octave = 0; octave < 2; octave++) {
            uint64_t low = GATE_OCTAVE_SALTS[climate][octave][0];
            uint64_t high = GATE_OCTAVE_SALTS[climate][octave][1];
            gateStep(&low, &high);
            steppedSalts[climate][octave][0] = low;
            steppedSalts[climate][octave][1] = high;
        }
    }
}

/**
 * @brief Returns the weighted gate terms for half of a climate value
 *
 * @param climate Which climate value (humidity, erosion then weirdness)
 * @param half Which half of the climate value
 * @param first The first random number for this half from the climate generator
 * @param second The second random number
 * @return uint32_t The total for this half
 */
static inline uint32_t halfTerms(int climate, int half, uint64_t first, uint64_t second) {
    gateStep(&first, &second); // skip the x offset
    return GATE_INT_WEIGHTS[climate][half][0] * gateTerm(first ^ steppedSalts[climate][0][0], second ^ steppedSalts[climate][0][1])
         + GATE_INT_WEIGHTS[climate][half][1] * gateTerm(first ^ steppedSalts[climate][1][0], second ^ steppedSalts[climate][1][1]);
}

/**
 * @brief Checks if a stream index passes the gate, with the cuts in the order gateKernel uses them
 *
 * @param cuts The gate cuts
 * @param index The stream index to check
 * @return int 1 if the index passes, 0 if it doesn't
 */
static int passesGate(GateCuts cuts, uint64_t index) {
    uint64_t low = mixBits(index * STREAM_GOLDEN);
    uint64_t high = mixBits((index + 1) * STREAM_GOLDEN);
    uint64_t seedLow = gateOutput(low, high);
    gateStep(&low, &high);
    uint64_t seedHigh = gateOutput(low, high);

    uint32_t total = 0;
    for (int climate = 0; climate < 3; climate++) {
        low = seedLow ^ GATE_SALTS[climate][0];
        high = seedHigh ^ GATE_SALTS[climate][1];
        for (int half = 0; half < 2; half++) {
            uint64_t first = gateOutput(low, high);
            gateStep(&low, &high);
            uint64_t second = gateOutput(low, high);
            gateStep(&low, &high);
            total += halfTerms(climate, half, first, second);

            if (climate == 0 && half == 0 && total > cuts.firstHalf) {
                return 0;
            }
            if (climate == 0 && half == 1 && total > cuts.humidity) {
                return 0;
            }
            if (climate == 1 && half == 1 && total > cuts.total) {
                return 0;
            }
        }
    }
    return total <= cuts.total;
}

/**
 * @brief This method runs the gate on the indexes in a plain loop
 *
 * @param cuts The gate cuts
 * @param firstIndex The stream index to start from
 * @param indexCount The number of indexes to check
 * @param output Where the indexes that pass get written, in order
 * @return size_t The number of indexes that passed
 */
static size_t gateOneByOne(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    size_t passCount = 0;
    for (size_t i = 0; i < indexCount; i++) {
        if (passesGate(cuts, firstIndex + i)) {
            output[passCount++] = firstIndex + i;
        }
    }
    return passCount;
}

// The AVX-512 version of the gate, it checks a vector of indexes in one go
#ifdef __AVX512DQ__
#include <immintrin.h>

#define BROADCAST(value) _mm512_set1_epi64((long long) (value))

// gateOutput, gateStep, mixBits for vectors of seeds
static inline __m512i outputVectors(__m512i low, __m512i high) {
    return _mm512_add_epi64(_mm512_rol_epi64(_mm512_add_epi64(low, high), 17), low);
}

static inline void stepVectors(__m512i *low, __m512i *high) {
    __m512i mixed = _mm512_xor_si512(*high, *low);
    *low = _mm512_ternarylogic_epi64(_mm512_rol_epi64(*low, 49), mixed, _mm512_slli_epi64(mixed, 21), 0x96);
    *high = _mm512_rol_epi64(mixed, 28);
}

static inline __m512i mixVectors(__m512i value) {
    value = _mm512_mullo_epi64(_mm512_xor_si512(value, _mm512_srli_epi64(value, 30)), BROADCAST(0xbf58476d1ce4e5b9ULL));
    value = _mm512_mullo_epi64(_mm512_xor_si512(value, _mm512_srli_epi64(value, 27)), BROADCAST(0x94d049bb133111ebULL));
    return _mm512_xor_si512(value, _mm512_srli_epi64(value, 31));
}

/**
 * @brief Returns weight * gateTerm for a vector of seeds (the terms with their weights fit in a 32 bit multiply)
 *
 * @param low The low halves of the octave generators
 * @param high The high halves
 * @param weight The octave's weight
 * @return __m512i The weighted terms
 */
static inline __m512i termVectors(__m512i low, __m512i high, uint32_t weight) {
    __m512i rotated = _mm512_rol_epi64(_mm512_add_epi64(low, high), 17);
    __m512i fraction = _mm512_and_si512(_mm512_add_epi64(_mm512_srli_epi64(rotated, 32), _mm512_srli_epi64(low, 32)), BROADCAST(0xFFFFFF));
    __m512i distance = _mm512_abs_epi64(_mm512_sub_epi64(fraction, BROADCAST(0x800000)));
    return _mm512_mul_epu32(_mm512_srli_epi64(distance, 7), BROADCAST(weight));
}

/**
 * @brief Returns the weighted gate terms for half of a climate value, for a vector of seeds (like halfTerms)
 *
 * @param climate Which climate value (humidity, erosion then weirdness)
 * @param half Which half of the climate value
 * @param first The first random numbers for this half from the climate generators
 * @param second The second random numbers
 * @return __m512i The totals for this half
 */
static inline __m512i halfTermVectors(int climate, int half, __m512i first, __m512i second) {
    stepVectors(&first, &second);
    __m512i octave0 = termVectors(_mm512_xor_si512(first, BROADCAST(steppedSalts[climate][0][0])),
                                  _mm512_xor_si512(second, BROADCAST(steppedSalts[climate][0][1])), GATE_INT_WEIGHTS[climate][half][0]);
    __m512i octave1 = termVectors(_mm512_xor_si512(first, BROADCAST(steppedSalts[climate][1][0])),
                                  _mm512_xor_si512(second, BROADCAST(steppedSalts[climate][1][1])), GATE_INT_WEIGHTS[climate][half][1]);
    return _mm512_add_epi64(octave0, octave1);
}

#define QUEUE_SIZE 4096 // How many indexes go through the gate together

size_t gateIndexes(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    // The seeds that are left after a stage: the index, the seed's random numbers, the humidity generator and the total so far
    static __thread uint64_t queueIndexes[QUEUE_SIZE + 16], queueLows[QUEUE_SIZE + 16], queueHighs[QUEUE_SIZE + 16];
    static __thread uint64_t queueClimateLows[QUEUE_SIZE + 16], queueClimateHighs[QUEUE_SIZE + 16], queueTotals[QUEUE_SIZE + 16];
    const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i firstHalfCut = BROADCAST(cuts.firstHalf);
    const __m512i humidityCut = BROADCAST(cuts.humidity);
    const __m512i totalCut = BROADCAST(cuts.total);

    size_t passCount = 0;
    size_t i = 0;
    while (i < indexCount) {
        size_t end = indexCount;
        if (i + QUEUE_SIZE < indexCount) {
            end = i + QUEUE_SIZE;
        }
        size_t vectorEnd = i + ((end - i) & ~7UL);
        size_t queued = 0;

        // Stage 1 is the first half of humidity. The mixes for the next group also give us the mix after an index. The next group's
        // products are this group's plus goldenStep (adding is cheaper than multiplying)
        __m512i golden = _mm512_mullo_epi64(_mm512_add_epi64(BROADCAST(firstIndex + i), lanes), BROADCAST(STREAM_GOLDEN));
        const __m512i goldenStep = BROADCAST(8 * STREAM_GOLDEN);
        __m512i mixed = mixVectors(golden);
        for (; i < vectorEnd; i += 8) {
            golden = _mm512_add_epi64(golden, goldenStep);
            __m512i nextMixed = mixVectors(golden);
            __m512i low = mixed;
            __m512i high = _mm512_alignr_epi64(nextMixed, mixed, 1);
            mixed = nextMixed;
            __m512i seedLow = outputVectors(low, high);
            stepVectors(&low, &high);
            __m512i seedHigh = outputVectors(low, high);

            low = _mm512_xor_si512(seedLow, BROADCAST(GATE_SALTS[0][0]));
            high = _mm512_xor_si512(seedHigh, BROADCAST(GATE_SALTS[0][1]));
            __m512i first = outputVectors(low, high);
            stepVectors(&low, &high);
            __m512i second = outputVectors(low, high);
            stepVectors(&low, &high);
            __m512i totals = halfTermVectors(0, 0, first, second);
            __mmask8 keep = _mm512_cmple_epu64_mask(totals, firstHalfCut);
            if (keep) {
                __m512i indexes = _mm512_add_epi64(BROADCAST(firstIndex + i), lanes);
                _mm512_storeu_si512(queueIndexes + queued, _mm512_maskz_compress_epi64(keep, indexes));
                _mm512_storeu_si512(queueLows + queued, _mm512_maskz_compress_epi64(keep, seedLow));
                _mm512_storeu_si512(queueHighs + queued, _mm512_maskz_compress_epi64(keep, seedHigh));
                _mm512_storeu_si512(queueClimateLows + queued, _mm512_maskz_compress_epi64(keep, low));
                _mm512_storeu_si512(queueClimateHighs + queued, _mm512_maskz_compress_epi64(keep, high));
                _mm512_storeu_si512(queueTotals + queued, _mm512_maskz_compress_epi64(keep, totals));
                queued += (size_t) __builtin_popcount(keep);
            }
        }

        // The indexes left at the end go through passesGate
        uint64_t leftovers[8];
        size_t leftoverCount = gateOneByOne(cuts, firstIndex + i, end - i, leftovers);
        i = end;

        // Stage 2 is the second half of humidity. We fill up the last group with seeds that can't pass
        for (size_t j = queued; j < ((queued + 7) & ~7UL); j++) {
            queueClimateHighs[j] = 0;
            queueClimateLows[j] = 0;
            queueHighs[j] = 0;
            queueLows[j] = 0;
            queueIndexes[j] = 0;
            queueTotals[j] = 0xFFFFFFFFu;
        }
        size_t kept = 0;
        for (size_t j = 0; j < queued; j += 8) {
            __m512i low = _mm512_loadu_si512(queueClimateLows + j);
            __m512i high = _mm512_loadu_si512(queueClimateHighs + j);
            __m512i first = outputVectors(low, high);
            stepVectors(&low, &high);
            __m512i second = outputVectors(low, high);
            __m512i totals = _mm512_add_epi64(_mm512_loadu_si512(queueTotals + j), halfTermVectors(0, 1, first, second));
            __mmask8 keep = _mm512_cmple_epu64_mask(totals, humidityCut);
            if (keep) {
                // kept can't get ahead of j, this writes over groups we already read
                _mm512_storeu_si512(queueIndexes + kept, _mm512_maskz_compress_epi64(keep, _mm512_loadu_si512(queueIndexes + j)));
                _mm512_storeu_si512(queueLows + kept, _mm512_maskz_compress_epi64(keep, _mm512_loadu_si512(queueLows + j)));
                _mm512_storeu_si512(queueHighs + kept, _mm512_maskz_compress_epi64(keep, _mm512_loadu_si512(queueHighs + j)));
                _mm512_storeu_si512(queueTotals + kept, _mm512_maskz_compress_epi64(keep, totals));
                kept += (size_t) __builtin_popcount(keep);
            }
        }
        queued = kept;

        // The last stages are erosion, then weirdness
        for (int climate = 1; climate <= 2; climate++) {
            for (size_t j = queued; j < ((queued + 7) & ~7UL); j++) {
                queueHighs[j] = 0;
                queueLows[j] = 0;
                queueIndexes[j] = 0;
                queueTotals[j] = 0xFFFFFFFFu;
            }
            kept = 0;
            for (size_t j = 0; j < queued; j += 8) {
                __m512i seedLow = _mm512_loadu_si512(queueLows + j);
                __m512i seedHigh = _mm512_loadu_si512(queueHighs + j);
                __m512i low = _mm512_xor_si512(seedLow, BROADCAST(GATE_SALTS[climate][0]));
                __m512i high = _mm512_xor_si512(seedHigh, BROADCAST(GATE_SALTS[climate][1]));
                __m512i totals = _mm512_loadu_si512(queueTotals + j);
                for (int half = 0; half < 2; half++) {
                    __m512i first = outputVectors(low, high);
                    stepVectors(&low, &high);
                    __m512i second = outputVectors(low, high);
                    stepVectors(&low, &high);
                    totals = _mm512_add_epi64(totals, halfTermVectors(climate, half, first, second));
                }
                __mmask8 keep = _mm512_cmple_epu64_mask(totals, totalCut);
                if (keep) {
                    _mm512_storeu_si512(queueIndexes + kept, _mm512_maskz_compress_epi64(keep, _mm512_loadu_si512(queueIndexes + j)));
                    if (climate == 1) {
                        _mm512_storeu_si512(queueLows + kept, _mm512_maskz_compress_epi64(keep, seedLow));
                        _mm512_storeu_si512(queueHighs + kept, _mm512_maskz_compress_epi64(keep, seedHigh));
                        _mm512_storeu_si512(queueTotals + kept, _mm512_maskz_compress_epi64(keep, totals));
                    }
                    kept += (size_t) __builtin_popcount(keep);
                }
            }
            queued = kept;
        }

        // The queue is in index order, followed by the leftovers, that keeps the output in order
        memcpy(output + passCount, queueIndexes, queued * 8);
        passCount += queued;
        for (size_t j = 0; j < leftoverCount; j++) {
            output[passCount++] = leftovers[j];
        }
    }
    return passCount;
}

int gateLanes(void) {
    return 8;
}
#elif defined(__AVX2__)
// The AVX2 version of the gate, it checks a vector of indexes in one go. AVX2 doesn't have the rotates, the long multiply or
// the compress the AVX-512 version uses, and this builds them out of the instructions it does have
#include <immintrin.h>

#define BROADCAST(value) _mm256_set1_epi64x((long long) (value))

// A rotate made out of a pair of shifts
static inline __m256i rotateVectors(__m256i value, int bits) {
    return _mm256_or_si256(_mm256_slli_epi64(value, bits), _mm256_srli_epi64(value, 64 - bits));
}

// Returns the low half of a * b, put together from smaller multiplies
static inline __m256i multiplyVectors(__m256i a, __m256i b) {
    __m256i aHigh = _mm256_srli_epi64(a, 32);
    __m256i bHigh = _mm256_srli_epi64(b, 32);
    __m256i lowProduct = _mm256_mul_epu32(a, b);
    __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(a, bHigh), _mm256_mul_epu32(aHigh, b));
    return _mm256_add_epi64(lowProduct, _mm256_slli_epi64(cross, 32));
}

// gateOutput, gateStep, mixBits for vectors of seeds
static inline __m256i outputVectors(__m256i low, __m256i high) {
    return _mm256_add_epi64(rotateVectors(_mm256_add_epi64(low, high), 17), low);
}

static inline void stepVectors(__m256i *low, __m256i *high) {
    __m256i mixed = _mm256_xor_si256(*high, *low);
    *low = _mm256_xor_si256(_mm256_xor_si256(rotateVectors(*low, 49), mixed), _mm256_slli_epi64(mixed, 21));
    *high = rotateVectors(mixed, 28);
}

static inline __m256i mixVectors(__m256i value) {
    value = multiplyVectors(_mm256_xor_si256(value, _mm256_srli_epi64(value, 30)), BROADCAST(0xbf58476d1ce4e5b9ULL));
    value = multiplyVectors(_mm256_xor_si256(value, _mm256_srli_epi64(value, 27)), BROADCAST(0x94d049bb133111ebULL));
    return _mm256_xor_si256(value, _mm256_srli_epi64(value, 31));
}

/**
 * @brief Returns weight * gateTerm for a vector of seeds, like the AVX-512 termVectors
 *
 * @param low The low halves of the octave generators
 * @param high The high halves
 * @param weight The octave's weight
 * @return __m256i The weighted terms
 */
static inline __m256i termVectors(__m256i low, __m256i high, uint32_t weight) {
    __m256i rotated = rotateVectors(_mm256_add_epi64(low, high), 17);
    __m256i fraction = _mm256_and_si256(_mm256_add_epi64(_mm256_srli_epi64(rotated, 32), _mm256_srli_epi64(low, 32)), BROADCAST(0xFFFFFF));
    // AVX2 doesn't have a long abs. The mask leaves the high half of the lanes at 0, that lets the short abs do the job
    __m256i distance = _mm256_abs_epi32(_mm256_sub_epi32(fraction, BROADCAST(0x800000)));
    return _mm256_mul_epu32(_mm256_srli_epi64(distance, 7), BROADCAST(weight));
}

/**
 * @brief Returns the weighted gate terms for half of a climate value, for a vector of seeds (like halfTerms)
 *
 * @param climate Which climate value (humidity, erosion then weirdness)
 * @param half Which half of the climate value
 * @param first The first random numbers for this half from the climate generators
 * @param second The second random numbers
 * @return __m256i The totals for this half
 */
static inline __m256i halfTermVectors(int climate, int half, __m256i first, __m256i second) {
    stepVectors(&first, &second);
    __m256i octave0 = termVectors(_mm256_xor_si256(first, BROADCAST(steppedSalts[climate][0][0])),
                                  _mm256_xor_si256(second, BROADCAST(steppedSalts[climate][0][1])), GATE_INT_WEIGHTS[climate][half][0]);
    __m256i octave1 = termVectors(_mm256_xor_si256(first, BROADCAST(steppedSalts[climate][1][0])),
                                  _mm256_xor_si256(second, BROADCAST(steppedSalts[climate][1][1])), GATE_INT_WEIGHTS[climate][half][1]);
    return _mm256_add_epi64(octave0, octave1);
}

// Returns a bit for the lanes with a total up to cut. The totals stay small and the filler seeds get 0xFFFFFFFF, that keeps the
// signed compare right
static inline int passingLanes(__m256i totals, __m256i cut) {
    return ~_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(totals, cut))) & 15;
}

// The compress from the AVX-512 version, as a permute. moves[lanes] packs the lanes in lanes to the front
static void loadCompressMoves(__m256i moves[16]) {
    for (int lanes = 0; lanes < 16; lanes++) {
        int32_t words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int slot = 0;
        for (int lane = 0; lane < 4; lane++) {
            if (lanes & (1 << lane)) {
                words[slot++] = lane + lane;
                words[slot++] = lane + lane + 1;
            }
        }
        moves[lanes] = _mm256_loadu_si256((const __m256i *) words);
    }
}

static inline void storeCompressed(uint64_t *destination, __m256i value, __m256i move) {
    _mm256_storeu_si256((__m256i *) destination, _mm256_permutevar8x32_epi32(value, move));
}

static inline __m256i loadVector(const uint64_t *source) {
    return _mm256_loadu_si256((const __m256i *) source);
}

#define QUEUE_SIZE 4096 // How many indexes go through the gate together

size_t gateIndexes(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    // The seeds that are left after a stage: the index, the seed's random numbers, the humidity generator and the total so far
    static __thread uint64_t queueIndexes[QUEUE_SIZE + 8], queueLows[QUEUE_SIZE + 8], queueHighs[QUEUE_SIZE + 8];
    static __thread uint64_t queueClimateLows[QUEUE_SIZE + 8], queueClimateHighs[QUEUE_SIZE + 8], queueTotals[QUEUE_SIZE + 8];
    const __m256i lanes = _mm256_set_epi64x(3, 2, 1, 0);
    const __m256i firstHalfCut = BROADCAST(cuts.firstHalf);
    const __m256i humidityCut = BROADCAST(cuts.humidity);
    const __m256i totalCut = BROADCAST(cuts.total);
    __m256i moves[16];
    loadCompressMoves(moves);

    size_t passCount = 0;
    size_t i = 0;
    while (i < indexCount) {
        size_t end = indexCount;
        if (i + QUEUE_SIZE < indexCount) {
            end = i + QUEUE_SIZE;
        }
        size_t vectorEnd = i + ((end - i) & ~3UL);
        size_t queued = 0;

        // Stage 1 is the first half of humidity. The mixes for the next group also give us the mix after an index. The next group's
        // products are this group's plus goldenStep (adding is cheaper than multiplying)
        __m256i golden = multiplyVectors(_mm256_add_epi64(BROADCAST(firstIndex + i), lanes), BROADCAST(STREAM_GOLDEN));
        const __m256i goldenStep = BROADCAST(4 * STREAM_GOLDEN);
        __m256i mixed = mixVectors(golden);
        for (; i < vectorEnd; i += 4) {
            golden = _mm256_add_epi64(golden, goldenStep);
            __m256i nextMixed = mixVectors(golden);
            // high gets the mix for the next index: the top lanes of mixed, then the bottom lane of nextMixed
            __m256i low = mixed;
            __m256i high = _mm256_blend_epi32(_mm256_permute4x64_epi64(mixed, _MM_SHUFFLE(0, 3, 2, 1)),
                                              _mm256_permute4x64_epi64(nextMixed, _MM_SHUFFLE(0, 3, 2, 1)), 0xC0);
            mixed = nextMixed;
            __m256i seedLow = outputVectors(low, high);
            stepVectors(&low, &high);
            __m256i seedHigh = outputVectors(low, high);

            low = _mm256_xor_si256(seedLow, BROADCAST(GATE_SALTS[0][0]));
            high = _mm256_xor_si256(seedHigh, BROADCAST(GATE_SALTS[0][1]));
            __m256i first = outputVectors(low, high);
            stepVectors(&low, &high);
            __m256i second = outputVectors(low, high);
            stepVectors(&low, &high);
            __m256i totals = halfTermVectors(0, 0, first, second);
            int passed = passingLanes(totals, firstHalfCut);
            if (passed) {
                __m256i indexes = _mm256_add_epi64(BROADCAST(firstIndex + i), lanes);
                storeCompressed(queueIndexes + queued, indexes, moves[passed]);
                storeCompressed(queueLows + queued, seedLow, moves[passed]);
                storeCompressed(queueHighs + queued, seedHigh, moves[passed]);
                storeCompressed(queueClimateLows + queued, low, moves[passed]);
                storeCompressed(queueClimateHighs + queued, high, moves[passed]);
                storeCompressed(queueTotals + queued, totals, moves[passed]);
                queued += (size_t) __builtin_popcount((unsigned) passed);
            }
        }

        // The indexes left at the end go through passesGate
        uint64_t leftovers[8];
        size_t leftoverCount = gateOneByOne(cuts, firstIndex + i, end - i, leftovers);
        i = end;

        // Stage 2 is the second half of humidity. We fill up the last group with seeds that can't pass
        for (size_t j = queued; j < ((queued + 3) & ~3UL); j++) {
            queueClimateHighs[j] = 0;
            queueClimateLows[j] = 0;
            queueHighs[j] = 0;
            queueLows[j] = 0;
            queueIndexes[j] = 0;
            queueTotals[j] = 0xFFFFFFFFu;
        }
        size_t kept = 0;
        for (size_t j = 0; j < queued; j += 4) {
            __m256i low = loadVector(queueClimateLows + j);
            __m256i high = loadVector(queueClimateHighs + j);
            __m256i first = outputVectors(low, high);
            stepVectors(&low, &high);
            __m256i second = outputVectors(low, high);
            __m256i totals = _mm256_add_epi64(loadVector(queueTotals + j), halfTermVectors(0, 1, first, second));
            int passed = passingLanes(totals, humidityCut);
            if (passed) {
                // kept can't get ahead of j, this writes over groups we already read
                storeCompressed(queueIndexes + kept, loadVector(queueIndexes + j), moves[passed]);
                storeCompressed(queueLows + kept, loadVector(queueLows + j), moves[passed]);
                storeCompressed(queueHighs + kept, loadVector(queueHighs + j), moves[passed]);
                storeCompressed(queueTotals + kept, totals, moves[passed]);
                kept += (size_t) __builtin_popcount((unsigned) passed);
            }
        }
        queued = kept;

        // The last stages are erosion, then weirdness
        for (int climate = 1; climate <= 2; climate++) {
            for (size_t j = queued; j < ((queued + 3) & ~3UL); j++) {
                queueHighs[j] = 0;
                queueLows[j] = 0;
                queueIndexes[j] = 0;
                queueTotals[j] = 0xFFFFFFFFu;
            }
            kept = 0;
            for (size_t j = 0; j < queued; j += 4) {
                __m256i seedLow = loadVector(queueLows + j);
                __m256i seedHigh = loadVector(queueHighs + j);
                __m256i low = _mm256_xor_si256(seedLow, BROADCAST(GATE_SALTS[climate][0]));
                __m256i high = _mm256_xor_si256(seedHigh, BROADCAST(GATE_SALTS[climate][1]));
                __m256i totals = loadVector(queueTotals + j);
                for (int half = 0; half < 2; half++) {
                    __m256i first = outputVectors(low, high);
                    stepVectors(&low, &high);
                    __m256i second = outputVectors(low, high);
                    stepVectors(&low, &high);
                    totals = _mm256_add_epi64(totals, halfTermVectors(climate, half, first, second));
                }
                int passed = passingLanes(totals, totalCut);
                if (passed) {
                    storeCompressed(queueIndexes + kept, loadVector(queueIndexes + j), moves[passed]);
                    if (climate == 1) {
                        storeCompressed(queueLows + kept, seedLow, moves[passed]);
                        storeCompressed(queueHighs + kept, seedHigh, moves[passed]);
                        storeCompressed(queueTotals + kept, totals, moves[passed]);
                    }
                    kept += (size_t) __builtin_popcount((unsigned) passed);
                }
            }
            queued = kept;
        }

        // The queue is in index order, followed by the leftovers, that keeps the output in order
        memcpy(output + passCount, queueIndexes, queued * 8);
        passCount += queued;
        for (size_t j = 0; j < leftoverCount; j++) {
            output[passCount++] = leftovers[j];
        }
    }
    return passCount;
}

int gateLanes(void) {
    return 4;
}
#else
// If there's no AVX-512 or AVX2 the gate goes through the indexes in a plain loop
size_t gateIndexes(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    return gateOneByOne(cuts, firstIndex, indexCount, output);
}

int gateLanes(void) {
    return 1;
}
#endif
