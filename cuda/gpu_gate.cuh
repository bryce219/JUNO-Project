// The gate from hostgate.c on the GPU, in integers. Humidity gets added up for all of the indexes, then erosion and
// weirdness for the seeds that passed humidity (they go in a shared queue first to keep the warps full)
#pragma once
#include "hostgate.h"

#define GATE_THREADS 256
#define GATE_ITEMS 8 // How many indexes in a row a thread checks, the stream lets it reuse a mix for the next index
#define GATE_ROUND 2 // How many of those go through the queue together

// The climate salts from the CPU gate
__constant__ uint64_t GATE_CLIMATE_SALTS[3][2];

// The octave salts from the CPU gate after a step forward. uploadGpuGate fills this in
__constant__ uint64_t GATE_STEPPED_SALTS[3][2][2];

// The weights from the CPU gate, [climate][half][octave]
__constant__ uint32_t GATE_WEIGHTS[3][2][2];

// Returns the number xoroshiro would give for this state
DEV uint64_t gateOutput(uint64_t low, uint64_t high) {
    return rotateLeft(low + high, 17) + low;
}

// A xoroshiro step without making a number
DEV void gateStep(uint64_t &low, uint64_t &high) {
    high ^= low;
    low = rotateLeft(low, 49) ^ high ^ (high << 21);
    high = rotateLeft(high, 28);
}

/**
 * @brief Returns the distance between an octave's y offset fraction and a half, like gateTerm in hostgate.c
 *
 * @param low The low half of the octave's generator (past the x offset)
 * @param high The high half
 * @return uint32_t The distance from a half
 */
DEV uint32_t gateTerm(uint64_t low, uint64_t high) {
    // The high half of the output, leaving out the carry from the low half (that barely changes the result)
    uint64_t sum = low + high;
    uint32_t outputHigh = __funnelshift_l((uint32_t) sum, (uint32_t) (sum >> 32), 17) + (uint32_t) (low >> 32);
    int32_t centered = (int32_t) (outputHigh & 0xFFFFFFu) - 0x800000;
    return (uint32_t) abs(centered) >> 7;
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
DEV uint32_t halfTerms(int climate, int half, uint64_t first, uint64_t second) {
    gateStep(first, second); // skip the x offset (the octave salts were stepped in uploadGpuGate)
    uint32_t total = 0;
#pragma unroll
    for (int octave = 0; octave < 2; octave++) {
        total += GATE_WEIGHTS[climate][half][octave] * gateTerm(first ^ GATE_STEPPED_SALTS[climate][octave][0], second ^ GATE_STEPPED_SALTS[climate][octave][1]);
    }
    return total;
}

/**
 * @brief Gives a slot in the shared queue to the lanes of a warp that passed
 *
 * @param keep Whether or not this lane passed
 * @param count The number of slots used so far
 * @return int The lane's slot, or -1 if it didn't pass
 */
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
    uint32_t offset; // The index, counting from the block's first index
    uint32_t total;  // The humidity total
    uint64_t seedLow, seedHigh;
};

/**
 * @brief This kernel runs the gate on a batch of stream indexes, and the indexes that pass get added to output (not in order)
 *
 * @param first The stream index to start at (with the custom seed offset added)
 * @param count The number of indexes to check
 * @param cuts The gate cuts (see GateCuts in hostgate.h)
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

        // Humidity for this thread's indexes, the ones that pass go in the queue
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
        int queueLength = queued;
        for (int base = 0; base < queueLength; base += GATE_THREADS) {
            int queueIndex = base + threadIdx.x;
            bool passes = false;
            uint64_t index = 0;
            if (queueIndex < queueLength) {
                GateSurvivor entry = queue[queueIndex];
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

            // Add the indexes that passed to output, with an atomic add for the warp
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
 * @brief This method copies the salts and weights to the GPU. The octave salts get stepped forward first, since a xoroshiro
 * step is just xors and shifts (stepping the salt and the seed's numbers separately and xoring them gives you the stepped xor)
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
