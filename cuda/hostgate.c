// CPU gate for the scanner (--cpu-gate and --cpu-assist). It does the same sums as gateKernel in gpu_gate.cuh, in the same order,
// so both keep exactly the same seeds.
// Good seeds mostly have humidity, erosion and weirdness octaves with a y offset fraction close to a half, so we add up how far these
// are from a half (times a weight) and keep the seeds with a small total.
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "hostgate.h"

static inline uint64_t rotateLeft(uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

// A xoroshiro number from a generator state, and a step (the same as xNextLong in cubiomes, split in two)
static inline uint64_t gateOutput(uint64_t low, uint64_t high) {
    return rotateLeft(low + high, 17) + low;
}

static inline void gateStep(uint64_t *low, uint64_t *high) {
    uint64_t mixed = *high ^ *low;
    *low = rotateLeft(*low, 49) ^ mixed ^ (mixed << 21);
    *high = rotateLeft(mixed, 28);
}

// The mix xSetSeed uses
static inline uint64_t mixBits(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// How far the y offset fraction is from a half, see GATE_SCALE
static inline uint32_t gateTerm(uint64_t low, uint64_t high) {
    // only the high word of the output, without the carry out of the low word, like gateTerm in gpu_gate.cuh
    uint32_t outputHigh = (uint32_t) (rotateLeft(low + high, 17) >> 32) + (uint32_t) (low >> 32);
    int32_t centered = (int32_t) (outputHigh & 0xFFFFFFu) - 0x800000;
    return (uint32_t) (centered < 0 ? -centered : centered) >> 7;
}

// The octave salts after one step, worked out when the program starts
static uint64_t steppedSalts[3][2][2];

__attribute__((constructor)) static void stepSalts(void) {
    for (int climate = 0; climate < 3; climate++) {
        for (int octave = 0; octave < 2; octave++) {
            uint64_t low = GATE_OCTAVE_SALTS[climate][octave][0], high = GATE_OCTAVE_SALTS[climate][octave][1];
            gateStep(&low, &high);
            steppedSalts[climate][octave][0] = low;
            steppedSalts[climate][octave][1] = high;
        }
    }
}

static inline uint32_t halfTerms(int climate, int half, uint64_t first, uint64_t second) {
    gateStep(&first, &second); // skip the x offset
    return GATE_INT_WEIGHTS[climate][half][0] * gateTerm(first ^ steppedSalts[climate][0][0], second ^ steppedSalts[climate][0][1])
         + GATE_INT_WEIGHTS[climate][half][1] * gateTerm(first ^ steppedSalts[climate][1][0], second ^ steppedSalts[climate][1][1]);
}

// true if index passes the gate, one index at a time (the stages in gateKernel's order)
static int passesGate(GateCuts cuts, uint64_t index) {
    uint64_t low = mixBits(index * STREAM_GOLDEN), high = mixBits((index + 1) * STREAM_GOLDEN);
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

static size_t gateOneByOne(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    size_t passCount = 0;
    for (size_t i = 0; i < indexCount; i++) {
        if (passesGate(cuts, firstIndex + i)) {
            output[passCount++] = firstIndex + i;
        }
    }
    return passCount;
}

// AVX-512 version of the gate, it does 8 indexes at a time
#ifdef __AVX512DQ__
#include <immintrin.h>

#define BROADCAST(value) _mm512_set1_epi64((long long) (value))

static inline __m512i outputs(__m512i low, __m512i high) {
    return _mm512_add_epi64(_mm512_rol_epi64(_mm512_add_epi64(low, high), 17), low);
}

static inline void steps(__m512i *low, __m512i *high) {
    __m512i mixed = _mm512_xor_si512(*high, *low);
    *low = _mm512_ternarylogic_epi64(_mm512_rol_epi64(*low, 49), mixed, _mm512_slli_epi64(mixed, 21), 0x96);
    *high = _mm512_rol_epi64(mixed, 28);
}

static inline __m512i mixes(__m512i value) {
    value = _mm512_mullo_epi64(_mm512_xor_si512(value, _mm512_srli_epi64(value, 30)), BROADCAST(0xbf58476d1ce4e5b9ULL));
    value = _mm512_mullo_epi64(_mm512_xor_si512(value, _mm512_srli_epi64(value, 27)), BROADCAST(0x94d049bb133111ebULL));
    return _mm512_xor_si512(value, _mm512_srli_epi64(value, 31));
}

// weight * term for 8 seeds (a term is under 2^17 and a weight under 2^10, so the 32 bit multiply is exact)
static inline __m512i terms(__m512i low, __m512i high, uint32_t weight) {
    __m512i rotated = _mm512_rol_epi64(_mm512_add_epi64(low, high), 17);
    __m512i fraction = _mm512_and_si512(_mm512_add_epi64(_mm512_srli_epi64(rotated, 32), _mm512_srli_epi64(low, 32)), BROADCAST(0xFFFFFF));
    __m512i distance = _mm512_abs_epi64(_mm512_sub_epi64(fraction, BROADCAST(0x800000)));
    return _mm512_mul_epu32(_mm512_srli_epi64(distance, 7), BROADCAST(weight));
}

static inline __m512i halfTermVectors(int climate, int half, __m512i first, __m512i second) {
    steps(&first, &second);
    __m512i octave0 = terms(_mm512_xor_si512(first, BROADCAST(steppedSalts[climate][0][0])), _mm512_xor_si512(second, BROADCAST(steppedSalts[climate][0][1])),
                            GATE_INT_WEIGHTS[climate][half][0]);
    __m512i octave1 = terms(_mm512_xor_si512(first, BROADCAST(steppedSalts[climate][1][0])), _mm512_xor_si512(second, BROADCAST(steppedSalts[climate][1][1])),
                            GATE_INT_WEIGHTS[climate][half][1]);
    return _mm512_add_epi64(octave0, octave1);
}

#define QUEUE_SIZE 4096 // indexes that go through the gate together

size_t gateIndexes(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    // The seeds left after each stage: the index, the seed's two numbers, the humidity generator and the total so far
    static __thread uint64_t queueIndexes[QUEUE_SIZE + 16], queueLows[QUEUE_SIZE + 16], queueHighs[QUEUE_SIZE + 16];
    static __thread uint64_t queueClimateLows[QUEUE_SIZE + 16], queueClimateHighs[QUEUE_SIZE + 16], queueTotals[QUEUE_SIZE + 16];
    const __m512i lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i firstHalfCut = BROADCAST(cuts.firstHalf), humidityCut = BROADCAST(cuts.humidity), totalCut = BROADCAST(cuts.total);

    size_t passCount = 0;
    size_t i = 0;
    while (i < indexCount) {
        size_t end = i + QUEUE_SIZE < indexCount ? i + QUEUE_SIZE : indexCount;
        size_t vectorEnd = i + ((end - i) & ~7UL);
        size_t queued = 0;

        // Stage 1, humidity's first half. The mixes of the next 8 indexes also give each index the mix after it
        // (index + 8) * GOLDEN is the last group's product plus 8 * GOLDEN, an add instead of a 64 bit multiply
        __m512i golden = _mm512_mullo_epi64(_mm512_add_epi64(BROADCAST(firstIndex + i), lanes), BROADCAST(STREAM_GOLDEN));
        const __m512i goldenStep = BROADCAST(8 * STREAM_GOLDEN);
        __m512i mixed = mixes(golden);
        for (; i < vectorEnd; i += 8) {
            golden = _mm512_add_epi64(golden, goldenStep);
            __m512i nextMixed = mixes(golden);
            __m512i low = mixed, high = _mm512_alignr_epi64(nextMixed, mixed, 1);
            mixed = nextMixed;
            __m512i seedLow = outputs(low, high);
            steps(&low, &high);
            __m512i seedHigh = outputs(low, high);

            low = _mm512_xor_si512(seedLow, BROADCAST(GATE_SALTS[0][0]));
            high = _mm512_xor_si512(seedHigh, BROADCAST(GATE_SALTS[0][1]));
            __m512i first = outputs(low, high);
            steps(&low, &high);
            __m512i second = outputs(low, high);
            steps(&low, &high);
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

        // The last few indexes of the range go one by one
        uint64_t leftovers[8];
        size_t leftoverCount = gateOneByOne(cuts, firstIndex + i, end - i, leftovers);
        i = end;

        // Stage 2, humidity's second half. Seeds that can't pass fill up the last group of 8
        for (size_t j = queued; j < ((queued + 7) & ~7UL); j++) {
            queueIndexes[j] = queueLows[j] = queueHighs[j] = queueClimateLows[j] = queueClimateHighs[j] = 0;
            queueTotals[j] = 0xFFFFFFFFu;
        }
        size_t kept = 0;
        for (size_t j = 0; j < queued; j += 8) {
            __m512i low = _mm512_loadu_si512(queueClimateLows + j), high = _mm512_loadu_si512(queueClimateHighs + j);
            __m512i first = outputs(low, high);
            steps(&low, &high);
            __m512i second = outputs(low, high);
            __m512i totals = _mm512_add_epi64(_mm512_loadu_si512(queueTotals + j), halfTermVectors(0, 1, first, second));
            __mmask8 keep = _mm512_cmple_epu64_mask(totals, humidityCut);
            if (keep) {
                // kept never passes j, so this only writes over groups that were already read
                _mm512_storeu_si512(queueIndexes + kept, _mm512_maskz_compress_epi64(keep, _mm512_loadu_si512(queueIndexes + j)));
                _mm512_storeu_si512(queueLows + kept, _mm512_maskz_compress_epi64(keep, _mm512_loadu_si512(queueLows + j)));
                _mm512_storeu_si512(queueHighs + kept, _mm512_maskz_compress_epi64(keep, _mm512_loadu_si512(queueHighs + j)));
                _mm512_storeu_si512(queueTotals + kept, _mm512_maskz_compress_epi64(keep, totals));
                kept += (size_t) __builtin_popcount(keep);
            }
        }
        queued = kept;

        // Stages 3 and 4, erosion then weirdness
        for (int climate = 1; climate <= 2; climate++) {
            for (size_t j = queued; j < ((queued + 7) & ~7UL); j++) {
                queueIndexes[j] = queueLows[j] = queueHighs[j] = 0;
                queueTotals[j] = 0xFFFFFFFFu;
            }
            kept = 0;
            for (size_t j = 0; j < queued; j += 8) {
                __m512i seedLow = _mm512_loadu_si512(queueLows + j), seedHigh = _mm512_loadu_si512(queueHighs + j);
                __m512i low = _mm512_xor_si512(seedLow, BROADCAST(GATE_SALTS[climate][0]));
                __m512i high = _mm512_xor_si512(seedHigh, BROADCAST(GATE_SALTS[climate][1]));
                __m512i totals = _mm512_loadu_si512(queueTotals + j);
                for (int half = 0; half < 2; half++) {
                    __m512i first = outputs(low, high);
                    steps(&low, &high);
                    __m512i second = outputs(low, high);
                    steps(&low, &high);
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

        // The queue is in index order and the leftovers come after it, so the output stays in order
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
#else
// Without AVX-512 the gate checks the indexes one by one
size_t gateIndexes(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output) {
    return gateOneByOne(cuts, firstIndex, indexCount, output);
}

int gateLanes(void) {
    return 1;
}
#endif
