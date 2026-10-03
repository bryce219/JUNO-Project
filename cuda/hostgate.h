#ifndef HOSTGATE_H
#define HOSTGATE_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// The gate threshold, around one in a hundred stream indexes pass with this
#define GATE_THRESHOLD 0.62184 // I see this number in my dreams

// The stream: index j is the seed (j * GOLDEN_RATIO) ^ SILVER_RATIO. xSetSeed undoes the xor and mixes j * GOLDEN_RATIO and
// (j + 1) * GOLDEN_RATIO, so the next index needs just one new mix
#define STREAM_GOLDEN 0x9e3779b97f4a7c15ULL
#define STREAM_SILVER 0x6a09e667f3bcc909ULL

// The salts of the climate values the gate checks (humidity, erosion, weirdness)
static const uint64_t GATE_SALTS[3][2] = {
    {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL},
    {0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL},
    {0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL}};

// md5 salts for the first two octaves of each one
static const uint64_t GATE_OCTAVE_SALTS[3][2][2] = {
    {{0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, {0xf11268128982754fULL, 0x257a1d670430b0aaULL}},
    {{0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL}, {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}},
    {{0xf11268128982754fULL, 0x257a1d670430b0aaULL}, {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL}}};

// Octave weights times 1000, [climate][half][octave]. I got them from a logistic regression
static const uint32_t GATE_INT_WEIGHTS[3][2][2] = {
    {{1000, 281}, {984, 202}},
    {{464, 245}, {486, 193}},
    {{197, 182}, {209, 198}}};

// A term is |fraction - 2^23| >> 7 for the 24 bit y offset fraction, so a total is (gate total) * 1000 * 2^17
#define GATE_SCALE (1000.0 * 131072.0)

// The cuts for the stages of the gate (see gpu_gate.cuh), as integer totals
typedef struct {
    uint32_t firstHalf; // humidity, first half
    uint32_t humidity;  // all of humidity
    uint32_t total;     // the gate threshold
} GateCuts;

/**
 * @brief Runs the gate on a range of stream indexes, the same way gateKernel does
 *
 * @param cuts The cuts
 * @param firstIndex The stream index to start from
 * @param indexCount The number of indexes to check
 * @param output Where the indexes that pass get written, in order (it needs room for indexCount indexes)
 * @return size_t The number of indexes that passed
 */
size_t gateIndexes(GateCuts cuts, uint64_t firstIndex, size_t indexCount, uint64_t *output);

// Seeds the gate checks at a time: 8 with AVX-512, 1 without it
int gateLanes(void);

#ifdef __cplusplus
}
#endif
#endif
