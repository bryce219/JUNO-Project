// Saves the gradient selectors of an octave for the temperature and humidity kernels. After that the permutation array
// is free to shuffle the next octave into
#pragma once

/**
 * @brief Saves the gradient selectors for every lattice cell in a square block of cells
 * 
 * @param permutation The shuffled permutation array
 * @param latticeY The y lattice index of the octave
 * @param startCellX x of the corner cell
 * @param startCellZ z of the corner cell
 * @param side Cells on a side of the block
 * @param selectors Output, 8 selectors for each cell
 */
DEV void cacheSelectors(const uint8_t *permutation, uint8_t latticeY, int startCellX, int startCellZ, int side, uint8_t *selectors) {
    for (int row = 0; row < side; row++) {
        for (int column = 0; column < side; column++) {
            uint8_t latticeX = (uint8_t) (startCellX + column);
            uint8_t latticeZ = (uint8_t) (startCellZ + row);

            // hash the corners
            uint8_t hashX0 = permutation[latticeX] + latticeY;
            uint8_t hashX1 = permutation[latticeX + 1] + latticeY;
            uint8_t hashX0Y0 = permutation[hashX0] + latticeZ;
            uint8_t hashX1Y0 = permutation[hashX1] + latticeZ;
            uint8_t hashX0Y1 = permutation[hashX0 + 1] + latticeZ;
            uint8_t hashX1Y1 = permutation[hashX1 + 1] + latticeZ;

            // every point inside the cell uses these
            uint8_t *cell = selectors + (row * side + column) * 8;
            cell[0] = permutation[hashX0Y0];
            cell[1] = permutation[hashX1Y0];
            cell[2] = permutation[hashX0Y1];
            cell[3] = permutation[hashX1Y1];
            cell[4] = permutation[hashX0Y0 + 1];
            cell[5] = permutation[hashX1Y0 + 1];
            cell[6] = permutation[hashX0Y1 + 1];
            cell[7] = permutation[hashX1Y1 + 1];
        }
    }
}

// Like samplePerlinOctave in biome.cuh, but it reads the saved selectors instead of the permutation array
DEV float samplePerlinCached(const OctaveHeader *octave, const uint8_t *selectors, int side, int startCellX, int startCellZ,
                             double noiseX, double noiseZ) {
    float fadeY = octave->yFade;
    noiseX += octave->offsetX;
    noiseZ += octave->offsetZ;
    double floorX = floor(noiseX);
    double floorZ = floor(noiseZ);
    noiseX -= floorX;
    noiseZ -= floorZ;

    // Find the cell in the block, then fade the fractions
    int column = (uint8_t) ((int) floorX - startCellX);
    int row = (uint8_t) ((int) floorZ - startCellZ);
    float fadeX = (float) fade(noiseX);
    float fadeZ = (float) fade(noiseZ);
    float fractionX = (float) noiseX;
    float fractionY = (float) octave->yFraction;
    float fractionZ = (float) noiseZ;

    const uint8_t *cell = selectors + (row * side + column) * 8;

    float corner1 = gradientDot(cell[0], fractionX, fractionY, fractionZ);
    float corner2 = gradientDot(cell[1], fractionX - 1, fractionY, fractionZ);
    float corner3 = gradientDot(cell[2], fractionX, fractionY - 1, fractionZ);
    float corner4 = gradientDot(cell[3], fractionX - 1, fractionY - 1, fractionZ);
    float corner5 = gradientDot(cell[4], fractionX, fractionY, fractionZ - 1);
    float corner6 = gradientDot(cell[5], fractionX - 1, fractionY, fractionZ - 1);
    float corner7 = gradientDot(cell[6], fractionX, fractionY - 1, fractionZ - 1);
    float corner8 = gradientDot(cell[7], fractionX - 1, fractionY - 1, fractionZ - 1);

    corner1 = interpolate(fadeX, corner1, corner2);
    corner3 = interpolate(fadeX, corner3, corner4);
    corner5 = interpolate(fadeX, corner5, corner6);
    corner7 = interpolate(fadeX, corner7, corner8);
    corner1 = interpolate(fadeY, corner1, corner3);
    corner5 = interpolate(fadeY, corner5, corner7);
    return interpolate(fadeZ, corner1, corner5);
}
