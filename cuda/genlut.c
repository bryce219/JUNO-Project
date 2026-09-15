// Makes the biome lookup table for the GPU scanner (lut263.bin and lut263.h), the Makefile runs this.
//
// The depth is fixed; the biome only comes from the other five climate values. If we cut those axes at the edges of the leaf
// boxes in the biome tree we get a grid of cells, and the biome doesn't change inside a cell.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "biomenoise.h"
#include "tables/btree263.h"

#define FIXED_DEPTH (-15000) // has to match FIXED_DEPTH in biome.cuh
#define MAX_CUTS 64          // Max cuts on one axis

static const int AXES[5] = {0, 1, 2, 3, 5}; // Temperature, humidity, continentalness, erosion and weirdness (4 is depth)
static const char *AXIS_NAMES[5] = {"T", "H", "C", "E", "W"};
static const int NODE_COUNT = (int) (sizeof(btree263_nodes) / sizeof(uint64_t));

// for qsort
static int compareInts(const void *a, const void *b) {
    int32_t first = *(const int32_t *) a, second = *(const int32_t *) b;
    if (first < second) {
        return -1;
    }
    if (first > second) {
        return 1;
    }
    return 0;
}

// Checks if a tree node is a leaf
static int isLeaf(uint64_t node) {
    return (uint32_t) (node >> 48) >= (uint32_t) NODE_COUNT;
}

/**
 * @brief This method finds the cuts on an axis (the low end, and one past the high end, of each leaf box)
 * 
 * @param axis The axis (see AXES)
 * @param cuts The cuts go here, sorted
 * @return int The number of cuts, or -1 if there are too many
 */
static int findCuts(int axis, int32_t cuts[MAX_CUTS]) {
    // Get the edges of every leaf box on this axis
    int32_t *edges = (int32_t *) malloc(2 * (size_t) NODE_COUNT * sizeof(int32_t));
    int edgeCount = 0;
    for (int i = 0; i < NODE_COUNT; i++) {
        if (!isLeaf(btree263_nodes[i])) {
            continue;
        }
        int boxIndex = (btree263_nodes[i] >> (8 * AXES[axis])) & 0xFF;
        edges[edgeCount++] = btree263_param[boxIndex][0];
        edges[edgeCount++] = btree263_param[boxIndex][1] + 1;
    }

    // sort them and skip repeats
    qsort(edges, edgeCount, sizeof(int32_t), compareInts);
    int cutCount = 0;
    for (int i = 0; i < edgeCount; i++) {
        if (i > 0 && edges[i] == edges[i - 1]) {
            continue;
        }
        if (cutCount == MAX_CUTS) {
            free(edges);
            return -1;
        }
        cuts[cutCount++] = edges[i];
    }
    free(edges);
    return cutCount;
}

/**
 * @brief Gives the leaf biomes an index, in the order they show up in the tree
 * @param biomeIds Filled with the biome ids
 * @param biomeIndex Gets set to the index of a biome id (-1 if the biome isn't in the tree)
 * @return int The biome count, or -1 if there are too many
 */
static int findBiomes(uint8_t biomeIds[64], int8_t biomeIndex[256]) {
    int biomeCount = 0;
    memset(biomeIndex, -1, 256);
    for (int i = 0; i < NODE_COUNT; i++) {
        if (!isLeaf(btree263_nodes[i])) {
            continue;
        }
        int biome = (btree263_nodes[i] >> 48) & 0xFF;
        if (biomeIndex[biome] >= 0) {
            continue;
        }
        if (biomeCount == 64) {
            return -1;
        }
        biomeIndex[biome] = (int8_t) biomeCount;
        biomeIds[biomeCount++] = (uint8_t) biome;
    }
    return biomeCount;
}

/**
 * @brief Writes the header file with the size of the grid, the cuts and the biome ids
 * @param cutCount Cuts on each axis
 * @param cellCount Cells in the whole grid
 * @return int 0, or 1 if the file couldn't be written
 */
static int writeHeader(int32_t cuts[5][MAX_CUTS], int cutCount[5], long cellCount, const uint8_t *biomeIds, int biomeCount) {
    FILE *file = fopen("lut263.h", "w");
    if (!file) {
        printf("Couldn't write lut263.h!\n");
        return 1;
    }

    fprintf(file, "// Made by genlut.c, the biome lookup table for Minecraft 26.3\n");
    fprintf(file, "#pragma once\n");
    fprintf(file, "#include <cstdint>\n");
    fprintf(file, "\n");
    fprintf(file, "#define LUT_NH %d\n", cutCount[1] + 1);
    fprintf(file, "#define LUT_NC %d\n", cutCount[2] + 1);
    fprintf(file, "#define LUT_NE %d\n", cutCount[3] + 1);
    fprintf(file, "#define LUT_NW %d\n", cutCount[4] + 1);
    fprintf(file, "#define LUT_CELLS %ldL\n", cellCount);

    for (int axis = 0; axis < 5; axis++) {
        fprintf(file, "#define LUT_NCUT_%s %d\n", AXIS_NAMES[axis], cutCount[axis]);
        fprintf(file, "static const int32_t lut_cut_%s[%d] = {", AXIS_NAMES[axis], cutCount[axis]);
        for (int i = 0; i < cutCount[axis]; i++) {
            if (i > 0) {
                fprintf(file, ",");
            }
            fprintf(file, "%d", cuts[axis][i]);
        }
        fprintf(file, "};\n");
    }

    // The ids after the last biome are just 0
    fprintf(file, "static const uint8_t lut_id[64] = {");
    for (int i = 0; i < 64; i++) {
        int biome = 0;
        if (i < biomeCount) {
            biome = biomeIds[i];
        }
        if (i > 0) {
            fprintf(file, ",");
        }
        fprintf(file, "%d", biome);
    }
    fprintf(file, "};\n");
    fclose(file);
    return 0;
}

int main(void) {
    // Find the cuts on every axis
    int32_t cuts[5][MAX_CUTS];
    int cutCount[5];
    long cellCount = 1;
    for (int axis = 0; axis < 5; axis++) {
        cutCount[axis] = findCuts(axis, cuts[axis]);
        if (cutCount[axis] < 0) {
            printf("There are too many cuts on axis %s!\n", AXIS_NAMES[axis]);
            return 1;
        }
        cellCount *= cutCount[axis] + 1;
    }

    uint8_t biomeIds[64];
    int8_t biomeIndex[256];
    int biomeCount = findBiomes(biomeIds, biomeIndex);
    if (biomeCount < 0) {
        printf("Too many biomes for the table.\n");
        return 1;
    }

    // Each cell gets the biome at its low corner, the first cell on an axis uses the lowest cut minus one
    // sig segven
    uint8_t *table = (uint8_t *) malloc((size_t) cellCount);
    uint64_t climate[6];
    climate[4] = (uint64_t) (int64_t) FIXED_DEPTH;
    for (long cell = 0; cell < cellCount; cell++) {
        // split the cell number into an index on each axis (weirdness is last)
        long remaining = cell;
        for (int axis = 4; axis >= 0; axis--) {
            int index = (int) (remaining % (cutCount[axis] + 1));
            remaining /= cutCount[axis] + 1;

            int64_t value = (int64_t) cuts[axis][0] - 1;
            if (index > 0) {
                value = (int64_t) cuts[axis][index - 1];
            }
            climate[AXES[axis]] = (uint64_t) value;
        }
        table[cell] = (uint8_t) biomeIndex[climateToBiome(MC_26_3, climate, NULL)];
    }

    // Save the table, then the header
    FILE *file = fopen("lut263.bin", "wb");
    if (!file) {
        printf("Couldn't write lut263.bin!\n");
        return 1;
    }
    size_t bytesWritten = fwrite(table, 1, (size_t) cellCount, file);
    fclose(file);
    if (bytesWritten != (size_t) cellCount) {
        printf("Couldn't write lut263.bin!\n");
        return 1;
    }
    if (writeHeader(cuts, cutCount, cellCount, biomeIds, biomeCount)) {
        return 1;
    }

    printf("Wrote lut263.bin (%ld bytes) and lut263.h\n", cellCount);
    return 0;
}
