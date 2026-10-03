// The gate on the GPU. It adds up the same 12 terms as v1.2 (humidity, erosion and weirdness, the two lowest octaves of both
// halves), in integers now, in two parts:
//   1. humidity, both halves, for every index (two cuts: the first half's terms, then all of humidity)
//   2. erosion, then weirdness, for the seeds left after humidity (about 28%), packed together in shared memory so the warps
//      stay full
// hostgate.c does exactly the same sums on the CPU (for --cpu-gate and --cpu-assist).
#pragma once
#include "hostgate.h"

#define GATE_THREADS 256
#define GATE_ITEMS 8 // indexes in a row that a thread starts with, the stream lets it reuse a mix for the next one
#define GATE_ROUND 2 // how many of them go through a round (the queue holds GATE_THREADS * GATE_ROUND seeds)

__constant__ uint64_t GATE_CLIMATE_SALTS[3][2];
__constant__ uint64_t GATE_STEPPED_SALTS[3][2][2]; // the octave salts after one step, see uploadGpuGate
__constant__ uint32_t GATE_WEIGHTS[3][2][2];

DEV uint64_t gateOutput(uint64_t low, uint64_t high) {
    return rotateLeft(low + high, 17) + low;
}

DEV void gateStep(uint64_t &low, uint64_t &high) {
    high ^= low;
    low = rotateLeft(low, 49) ^ high ^ (high << 21);
    high = rotateLeft(high, 28);
}

// The distance of an octave's y offset fraction from a half, from its generator already past the x offset (see gateTerm in hostgate.c)
DEV uint32_t gateTerm(uint64_t low, uint64_t high) {
    // only the high word of the output, without the carry out of the low word (it moves the fraction by at most one unit)
    uint64_t sum = low + high;
    uint32_t outputHigh = __funnelshift_l((uint32_t) sum, (uint32_t) (sum >> 32), 17) + (uint32_t) (low >> 32);
    int32_t centered = (int32_t) (outputHigh & 0xFFFFFFu) - 0x800000;
    return (uint32_t) abs(centered) >> 7;
}

// The two terms for one half of a climate value. first and second are the half's two numbers from the climate generator
DEV uint32_t halfTerms(int climate, int half, uint64_t first, uint64_t second) {
    gateStep(first, second); // skip the x offset, the octave salts are stepped already
    uint32_t total = 0;
#pragma unroll
    for (int octave = 0; octave < 2; octave++) {
        total += GATE_WEIGHTS[climate][half][octave] * gateTerm(first ^ GATE_STEPPED_SALTS[climate][octave][0], second ^ GATE_STEPPED_SALTS[climate][octave][1]);
    }
    return total;
}

// Gives the lanes that keep going a slot in a shared queue, returns the slot (-1 for the others)
DEV int queueSlot(bool keep, int *count) {
    unsigned lanes = __ballot_sync(0xFFFFFFFFu, keep);
    int lane = threadIdx.x & 31;
    int leader = lanes ? __ffs(lanes) - 1 : 0;
    int first = 0;
    if (lanes && lane == leader) {
        first = atomicAdd(count, __popc(lanes));
    }
    first = __shfl_sync(0xFFFFFFFFu, first, leader);
    return keep ? first + __popc(lanes & ((1u << lane) - 1)) : -1;
}

// A seed that passed humidity, waiting for erosion and weirdness
struct GateSurvivor {
    uint32_t offset; // index minus the block's first index
    uint32_t total;  // humidity's terms
    uint64_t seedLow, seedHigh;
};

/**
 * @brief This kernel runs the gate on count stream indexes from first on. The ones that pass get added to output (not in order)
 *
 * @param first The stream index to start at (with the custom seed offset added)
 * @param count The number of indexes to check
 * @param cuts The cuts for each stage
 * @param output Where the indexes that pass go
 * @param outputCount The number of indexes that passed (it keeps going up past capacity)
 * @param capacity Room in output
 */
__global__ void __launch_bounds__(GATE_THREADS)
gateKernel(uint64_t first, uint32_t count, GateCuts cuts, uint64_t *output, uint32_t *outputCount, uint32_t capacity) {
    __shared__ GateSurvivor queue[GATE_THREADS * GATE_ROUND];
    __shared__ int queued;
    const uint32_t blockFirst = blockIdx.x * (GATE_THREADS * GATE_ITEMS);
    const uint32_t threadFirst = blockFirst + threadIdx.x * GATE_ITEMS;
    uint64_t nextMix = mixStream(first + threadFirst);

#pragma unroll 1
    for (int round = 0; round < GATE_ITEMS / GATE_ROUND; round++) {
        if (threadIdx.x == 0) {
            queued = 0;
        }
        __syncthreads();

#pragma unroll
        for (int item = 0; item < GATE_ROUND; item++) {
            uint32_t i = threadFirst + round * GATE_ROUND + item;
            uint64_t mix = nextMix;
            nextMix = mixStream(first + i + 1);
            bool keep = false;
            GateSurvivor entry;
            if (i < count) {
                seedNumbers(mix, nextMix, entry.seedLow, entry.seedHigh);
                uint64_t low = entry.seedLow ^ GATE_CLIMATE_SALTS[0][0], high = entry.seedHigh ^ GATE_CLIMATE_SALTS[0][1];
                uint64_t firstNumber = gateOutput(low, high);
                gateStep(low, high);
                uint64_t secondNumber = gateOutput(low, high);
                gateStep(low, high);
                uint32_t firstHalf = halfTerms(0, 0, firstNumber, secondNumber);
                firstNumber = gateOutput(low, high);
                gateStep(low, high);
                secondNumber = gateOutput(low, high);
                entry.total = firstHalf + halfTerms(0, 1, firstNumber, secondNumber);
                entry.offset = i - blockFirst;
                keep = firstHalf <= cuts.firstHalf && entry.total <= cuts.humidity;
            }
            int slot = queueSlot(keep, &queued);
            if (keep) {
                queue[slot] = entry;
            }
        }
        __syncthreads();

        // Erosion then weirdness for the seeds in the queue
        int stageCount = queued;
        for (int base = 0; base < stageCount; base += GATE_THREADS) {
            int q = base + threadIdx.x;
            bool passes = false;
            uint64_t index = 0;
            if (q < stageCount) {
                GateSurvivor entry = queue[q];
                index = first + blockFirst + entry.offset;
                uint32_t total = entry.total;
#pragma unroll
                for (int climate = 1; climate < 3; climate++) {
                    uint64_t low = entry.seedLow ^ GATE_CLIMATE_SALTS[climate][0], high = entry.seedHigh ^ GATE_CLIMATE_SALTS[climate][1];
#pragma unroll
                    for (int half = 0; half < 2; half++) {
                        uint64_t firstNumber = gateOutput(low, high);
                        gateStep(low, high);
                        uint64_t secondNumber = gateOutput(low, high);
                        gateStep(low, high);
                        total += halfTerms(climate, half, firstNumber, secondNumber);
                    }
                    if (total > cuts.total) {
                        break;
                    }
                }
                passes = total <= cuts.total;
            }
            unsigned passedLanes = __ballot_sync(0xFFFFFFFFu, passes);
            if (passedLanes) {
                int lane = threadIdx.x & 31;
                uint32_t warpFirst = 0;
                if (lane == 0) {
                    warpFirst = atomicAdd(outputCount, (uint32_t) __popc(passedLanes));
                }
                warpFirst = __shfl_sync(0xFFFFFFFFu, warpFirst, 0);
                uint32_t slot = warpFirst + __popc(passedLanes & ((1u << lane) - 1));
                if (passes && slot < capacity) {
                    output[slot] = index;
                }
            }
        }
        __syncthreads();
    }
}

/**
 * @brief This method copies the salts and weights to the GPU. The octave salts get one xoroshiro step first (a step is just xors
 * and shifts, so stepping the salt and the half's numbers seperately and xoring them is the same as stepping the xor)
 */
static void uploadGpuGate() {
    uint64_t stepped[3][2][2];
    for (int climate = 0; climate < 3; climate++) {
        for (int octave = 0; octave < 2; octave++) {
            uint64_t low = GATE_OCTAVE_SALTS[climate][octave][0], high = GATE_OCTAVE_SALTS[climate][octave][1];
            high ^= low;
            low = ((low << 49) | (low >> 15)) ^ high ^ (high << 21);
            high = (high << 28) | (high >> 36);
            stepped[climate][octave][0] = low;
            stepped[climate][octave][1] = high;
        }
    }
    cudaMemcpyToSymbol(GATE_STEPPED_SALTS, stepped, sizeof(stepped));
    cudaMemcpyToSymbol(GATE_CLIMATE_SALTS, GATE_SALTS, sizeof(GATE_SALTS));
    cudaMemcpyToSymbol(GATE_WEIGHTS, GATE_INT_WEIGHTS, sizeof(GATE_INT_WEIGHTS));
}
