// Gets the biome for a climate point from the lookup table genlut.c makes (a lot faster than going through the biome tree)
#pragma once
#include <cstdint>
#include "lut263.h"

// The cuts for the five axes, uploadLut copies them to the GPU
__constant__ int32_t TEMPERATURE_CUTS[LUT_NCUT_T];
__constant__ int32_t HUMIDITY_CUTS[LUT_NCUT_H];
__constant__ int32_t CONTINENTALNESS_CUTS[LUT_NCUT_C];
__constant__ int32_t EROSION_CUTS[LUT_NCUT_E];
__constant__ int32_t WEIRDNESS_CUTS[LUT_NCUT_W];

// counts the cuts that are <= value
template <int CUT_COUNT>
__device__ __forceinline__ int countCuts(const int32_t *cuts, int value) {
    int count = 0;
#pragma unroll
    for (int i = 0; i < CUT_COUNT; i++) {
        count += (value >= cuts[i]);
    }
    return count;
}

__device__ __forceinline__ int clampClimate(int64_t value) {
    if (value > 1000000) {
        return 1000000;
    }
    if (value < -1000000) {
        return -1000000;
    }
    return (int) value;
}

/**
 * @brief Returns the biome index for a climate point
 * 
 * @param table The lookup table on the GPU
 * @param climate The six climate values
 * @return uint8_t The biome index (lut_id has the biome id)
 */
__device__ __forceinline__ uint8_t lutCell(const uint8_t *table, const int64_t climate[6]) {
    int temperatureIndex = countCuts<LUT_NCUT_T>(TEMPERATURE_CUTS, clampClimate(climate[0]));
    int humidityIndex = countCuts<LUT_NCUT_H>(HUMIDITY_CUTS, clampClimate(climate[1]));
    int continentalnessIndex = countCuts<LUT_NCUT_C>(CONTINENTALNESS_CUTS, clampClimate(climate[2]));
    int erosionIndex = countCuts<LUT_NCUT_E>(EROSION_CUTS, clampClimate(climate[3]));
    int weirdnessIndex = countCuts<LUT_NCUT_W>(WEIRDNESS_CUTS, clampClimate(climate[5]));

    // Cells go in temperature, humidity, continentalness, erosion, weirdness order (the order genlut.c fills them in)
    long index = ((((long) temperatureIndex * LUT_NH + humidityIndex) * LUT_NC + continentalnessIndex) * LUT_NE + erosionIndex) * LUT_NW + weirdnessIndex;
    return __ldg(table + index);
}

// This part runs on the CPU
#include <cstdio>
#include <cstdlib>
#include <vector>

/**
 * This method loads the lookup table from the file and copies it to the GPU along with the cuts.
 * @param path Path to the lookup file
 * @return uint8_t* The lookup table on the GPU
 */
static uint8_t *uploadLut(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open %s, run make first and start the scanner from the cuda folder!\n", path);
        exit(1);
    }

    std::vector<uint8_t> hostTable(LUT_CELLS);
    size_t bytesRead = fread(hostTable.data(), 1, LUT_CELLS, file);
    fclose(file);
    if (bytesRead != (size_t) LUT_CELLS) {
        fprintf(stderr, "%s is too short, it should be the full lookup table!\n", path);
        exit(1);
    }

    uint8_t *deviceTable;
    if (cudaMalloc(&deviceTable, LUT_CELLS) != cudaSuccess) {
        fprintf(stderr, "Couldn't make room for the lookup table on the GPU\n");
        exit(1);
    }
    cudaMemcpy(deviceTable, hostTable.data(), LUT_CELLS, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(TEMPERATURE_CUTS, lut_cut_T, sizeof(lut_cut_T));
    cudaMemcpyToSymbol(HUMIDITY_CUTS, lut_cut_H, sizeof(lut_cut_H));
    cudaMemcpyToSymbol(CONTINENTALNESS_CUTS, lut_cut_C, sizeof(lut_cut_C));
    cudaMemcpyToSymbol(EROSION_CUTS, lut_cut_E, sizeof(lut_cut_E));
    cudaMemcpyToSymbol(WEIRDNESS_CUTS, lut_cut_W, sizeof(lut_cut_W));
    return deviceTable;
}
