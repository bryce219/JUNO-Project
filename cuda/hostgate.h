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

// Returns 1 if the gate was built with AVX-512, 0 if it uses the slow version
int gateUsesAvx512(void);

#ifdef __cplusplus
}
#endif
#endif
