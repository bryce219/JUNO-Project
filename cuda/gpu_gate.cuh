// The gate from hostgate.c, on the GPU and in floats
#pragma once

// The climate salts from hostgate.c
__constant__ uint64_t GATE_CLIMATE_SALTS[3][2] = {
    {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL},  // Humidity
    {0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL},  // Erosion
    {0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL}}; // Weirdness

// The octave salts from hostgate.c, already stepped once (a step is just xors and shifts). uploadGpuGate fills this in
__constant__ uint64_t GATE_STEPPED_SALTS[3][2][2];

// The weights from hostgate.c as floats, [climate][half][octave]
__constant__ float GATE_WEIGHTS[3][2][2] = {
    {{1.000f, 0.281f}, {0.984f, 0.202f}},  // Humidity
    {{0.464f, 0.245f}, {0.486f, 0.193f}},  // Erosion
    {{0.197f, 0.182f}, {0.209f, 0.198f}}}; // Weirdness

#define GATE_HUMIDITY_LIMIT 0.50 // We use hostgate.c's humidity cut too
#define GATE_THREADS 256
#define GATE_ITEMS 4             // How many indexes a thread checks humidity for

// A xoroshiro step without making a number
DEV void xoroshiroStep(uint64_t &low, uint64_t &high) {
    high ^= low;
    low = rotateLeft(low, 49) ^ high ^ (high << 21);
    high = rotateLeft(high, 28);
}

/**
 * @brief Returns the gate total for a climate value (times 2^24), like addClimate in hostgate.c
 * 
 * @param climate 0, 1 or 2 (humidity, erosion or weirdness)
 * @param seedLow The low random number of the seed
 * @param seedHigh The high one
 * @return float The total for this climate value
 */
DEV float gateClimate(int climate, uint64_t seedLow, uint64_t seedHigh) {
    XoroshiroState climateRandom;
    climateRandom.low = seedLow ^ GATE_CLIMATE_SALTS[climate][0];
    climateRandom.high = seedHigh ^ GATE_CLIMATE_SALTS[climate][1];
    float total = 0;

#pragma unroll
    for (int half = 0; half < 2; half++) {
        uint64_t low = xNextLong(&climateRandom);
        uint64_t high = xNextLong(&climateRandom);
        xoroshiroStep(low, high); // skip the x offset

#pragma unroll
        for (int octave = 0; octave < 2; octave++) {
            uint64_t octaveLow = low ^ GATE_STEPPED_SALTS[climate][octave][0];
            uint64_t octaveHigh = high ^ GATE_STEPPED_SALTS[climate][octave][1];
            uint64_t yRandom = rotateLeft(octaveLow + octaveHigh, 17) + octaveLow; // xNextLong, without the step after it
            // how far the y offset's fraction is from a half
            int distance = abs((int) ((uint32_t) (yRandom >> 32) & 0xFFFFFFu) - 0x800000);
            total += GATE_WEIGHTS[climate][half][octave] * (float) distance;
        }
    }
    return total;
}

// A seed that passed humidity, waiting for the erosion and weirdness part
struct GateQueueEntry {
    uint64_t index;
    uint64_t seedLow;
    uint64_t seedHigh;
    float total;
};

/**
 * @brief This kernel runs the gate on a batch of stream indexes, the indexes that pass get added to output (not in order).
 * The seeds that pass humidity go in a queue, then the block does erosion and weirdness for the queue
 * 
 * @param first The stream index to start at (with the custom seed offset added)
 * @param count The number of indexes to check
 * @param threshold The gate threshold times 2^24
 * @param humidityThreshold The humidity cut times 2^24
 * @param output Where the indexes that pass go
 * @param outputCount The number of indexes that passed (it keeps going up past capacity)
 * @param capacity Room in output
 */
__global__ void __launch_bounds__(GATE_THREADS)
gateKernel(uint64_t first, uint32_t count, float threshold, float humidityThreshold, uint64_t *output, uint32_t *outputCount, uint32_t capacity) {
    __shared__ GateQueueEntry queue[GATE_THREADS * GATE_ITEMS];
    __shared__ int queued;
    if (threadIdx.x == 0) {
        queued = 0;
    }
    __syncthreads();

    // Humidity for all of the indexes
    uint32_t blockFirst = blockIdx.x * (GATE_THREADS * GATE_ITEMS);
#pragma unroll
    for (int item = 0; item < GATE_ITEMS; item++) {
        uint32_t i = blockFirst + item * GATE_THREADS + threadIdx.x;
        if (i < count) {
            uint64_t index = first + i;
            XoroshiroState random;
            xSetSeed(&random, streamSeed(index));
            uint64_t seedLow = xNextLong(&random);
            uint64_t seedHigh = xNextLong(&random);
            float total = gateClimate(0, seedLow, seedHigh);
            if (total <= humidityThreshold) {
                int slot = atomicAdd(&queued, 1);
                queue[slot].index = index;
                queue[slot].seedLow = seedLow;
                queue[slot].seedHigh = seedHigh;
                queue[slot].total = total;
            }
        }
    }
    __syncthreads();

    // Then erosion and weirdness for the queue
    int queueLength = queued;
    for (int base = 0; base < queueLength; base += GATE_THREADS) {
        int queueIndex = base + threadIdx.x;
        bool passes = false;
        uint64_t index = 0;
        if (queueIndex < queueLength) {
            GateQueueEntry entry = queue[queueIndex];
            float total = entry.total + gateClimate(1, entry.seedLow, entry.seedHigh);
            if (total <= threshold) {
                total += gateClimate(2, entry.seedLow, entry.seedHigh);
            }
            passes = total <= threshold;
            index = entry.index;
        }

        // Lane zero gets room in output for the warp, then the threads that passed write their indexes
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
}

/**
 * @brief This method does a xoroshiro step on the octave salts and copies the results to the GPU (for GATE_STEPPED_SALTS)
 */
static void uploadGpuGate() {
    const uint64_t octaveSalts[3][2][2] = {
        {{0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, {0xf11268128982754fULL, 0x257a1d670430b0aaULL}},  // Humidity
        {{0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL}, {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}},  // Erosion
        {{0xf11268128982754fULL, 0x257a1d670430b0aaULL}, {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL}}}; // Weirdness
    uint64_t stepped[3][2][2];

    for (int climate = 0; climate < 3; climate++) {
        for (int octave = 0; octave < 2; octave++) {
            uint64_t low = octaveSalts[climate][octave][0], high = octaveSalts[climate][octave][1];
            high ^= low;
            low = ((low << 49) | (low >> 15)) ^ high ^ (high << 21);
            high = (high << 28) | (high >> 36);
            stepped[climate][octave][0] = low;
            stepped[climate][octave][1] = high;
        }
    }
    cudaMemcpyToSymbol(GATE_STEPPED_SALTS, stepped, sizeof(stepped));
}
