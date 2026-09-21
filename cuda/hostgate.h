#ifndef HOSTGATE_H
#define HOSTGATE_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The gate threshold, around one in a hundred stream indexes pass with this
#define GATE_THRESHOLD 0.62184 // I see this number in my dreams

/**
 * @brief Runs the gate on a range of stream indexes
 * 
 * @param threshold The gate threshold (GATE_THRESHOLD)
 * @param firstIndex The stream index to start from
 * @param indexCount The number of indexes to check
 * @param output Where the indexes that pass get written, in order (it needs room for indexCount indexes)
 * @return size_t The number of indexes that passed
 */
size_t gateIndexes(double threshold, uint64_t firstIndex, size_t indexCount, uint64_t *output);

/**
 * @brief Runs the gate in floats, the way the GPU gate does (for --cpu-assist). The rest works like gateIndexes
 * 
 * @param threshold The GPU gate's threshold (gpuGateThreshold in scan.cu)
 * @param humidityThreshold The GPU gate's humidity cut (gpuGateHumidityThreshold)
 */
size_t floatGateIndexes(float threshold, float humidityThreshold, uint64_t firstIndex, size_t indexCount, uint64_t *output);

// Seeds the gate checks at a time: 8 with AVX-512, 4 with AVX2, 1 without either
int gateLanes(void);

// The same for floatGateIndexes, which needs FMA too
int floatGateLanes(void);

#ifdef __cplusplus
}
#endif
#endif
