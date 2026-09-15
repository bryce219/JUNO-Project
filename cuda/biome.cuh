// Climate noise for Minecraft 26.3 on the GPU, ported from cubiomes (MIT license).
// We read the biomes above ground with a fixed depth, so the depth spline isn't needed at all.
#pragma once
#include <cstdint>

// Device functions that get inlined into the kernels
#define DEV __device__ __forceinline__

// The xoroshiro random generator state (the one Minecraft uses)
struct XoroshiroState {
    uint64_t low;
    uint64_t high;
};

// The xoroshiro functions are the same as the ones in cubiomes rng.h
DEV uint64_t rotateLeft(uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

/**
 * @brief Seeds the random generator the way Minecraft does
 * 
 * @param random The generator we want to seed
 * @param seed 
 */
DEV void xSetSeed(XoroshiroState *random, uint64_t seed) {
    const uint64_t GOLDEN_RATIO = 0x9e3779b97f4a7c15ULL;
    const uint64_t SILVER_RATIO = 0x6a09e667f3bcc909ULL;
    const uint64_t MIX_1 = 0xbf58476d1ce4e5b9ULL;
    const uint64_t MIX_2 = 0x94d049bb133111ebULL;

    uint64_t low = seed ^ SILVER_RATIO;
    uint64_t high = low + GOLDEN_RATIO;

    low = (low ^ (low >> 30)) * MIX_1;
    high = (high ^ (high >> 30)) * MIX_1;
    low = (low ^ (low >> 27)) * MIX_2;
    high = (high ^ (high >> 27)) * MIX_2;

    random->low = low ^ (low >> 31);
    random->high = high ^ (high >> 31);
}

DEV uint64_t xNextLong(XoroshiroState *random) {
    uint64_t low = random->low, high = random->high;
    uint64_t result = rotateLeft(low + high, 17) + low;

    high ^= low;
    random->low = rotateLeft(low, 49) ^ high ^ (high << 21);
    random->high = rotateLeft(high, 28);
    return result;
}

/**
 * @brief Returns a random int from 0 up to (but not including) the bound
 * 
 * @param random The generator to use
 * @param bound The upper bound
 * @return int The random int
 */
DEV int xNextInt(XoroshiroState *random, uint32_t bound) {
    uint64_t value = (xNextLong(random) & 0xFFFFFFFFu) * bound;
    if ((uint32_t) value < bound) {
        // Throw away values that would make the result uneven
        while ((uint32_t) value < (~bound + 1) % bound) {
            value = (xNextLong(random) & 0xFFFFFFFFu) * bound;
        }
    }
    return (int) (value >> 32);
}

DEV double xNextDouble(XoroshiroState *random) {
    return (xNextLong(random) >> 11) * 1.1102230246251565E-16; // 1 / 2^53
}

// Octaves for all six climate values (shift, temperature, humidity, continentalness, erosion and weirdness)
#define OCTAVE_COUNT 46

// A Perlin noise octave
struct Octave {
    uint8_t permutation[257];
    uint8_t yLattice;
    double offsetX;
    double offsetZ;
    double amplitude;
    double lacunarity;
    double yFraction;
    double yFade;
};

// An Octave without the permutation array. The temperature, humidity and probe kernels use this and keep the permutations seperate
struct OctaveHeader {
    double offsetX;
    double offsetZ;
    double amplitude;
    double lacunarity;
    double yFraction;
    double yFade;
    uint8_t yLattice;
};

// md5 salts of "octave_-12" to "octave_0"
__constant__ uint64_t OCTAVE_SALTS[13][2] = {
    {0xb198de63a8012672ULL, 0x7b84cad43ef7b5a8ULL}, {0x0fd787bfbc403ec3ULL, 0x74a4a31ca21b48b8ULL},
    {0x36d326eed40efeb2ULL, 0x5be9ce18223c636aULL}, {0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL},
    {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, {0xf11268128982754fULL, 0x257a1d670430b0aaULL},
    {0xe51c98ce7d1de664ULL, 0x5f9478a733040c45ULL}, {0x6d7b49e7e429850aULL, 0x2e3063c622a24777ULL},
    {0xbd90d5377ba1b762ULL, 0xc07317d419a7548dULL}, {0x53d39c6752dac858ULL, 0xbcd1c5a80ab65b3eULL},
    {0xb4a24d7a84e7677bULL, 0x023ff9668e89b5c4ULL}, {0xdffa22b534c5f608ULL, 0xb9b67517d3665ca9ULL},
    {0xd50708086cef4d7cULL, 0x6e1651ecc7f43309ULL}};

// The noise settings for a climate value
struct ClimateNoise {
    uint64_t saltLow;
    uint64_t saltHigh;
    int firstOctave;      // lowest octave (omin in cubiomes)
    int amplitudeCount;
    double amplitudes[9];
    double amplitude;     // amplitude of the double Perlin noise
    int firstSlot;        // Where this value's octaves start in the octave array
    int slotCount;        // octaves it has in the array
};

__constant__ ClimateNoise CLIMATE_NOISE[6] = {
    {0x080518cf6af25384ULL, 0x3f3dfb40a54febd5ULL, -3, 4, {1, 1, 1, 0}, 15.0 / 12, 0, 6},                     // Shift
    {0x5c7e6b29735f0d7fULL, 0xf7d86f1bbc734988ULL, -10, 6, {1.5, 0, 1, 0, 0, 0}, 15.0 / 12, 6, 4},            // Temperature
    {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL, -8, 6, {1, 1, 0, 0, 0, 0}, 10.0 / 9, 10, 4},               // Humidity
    {0x83886c9d0ae3a662ULL, 0xafa638a61b42e8adULL, -9, 9, {1, 1, 2, 2, 2, 1, 1, 1, 1}, 45.0 / 30, 14, 18},    // Continentalness
    {0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL, -9, 5, {1, 1, 0, 1, 1}, 25.0 / 18, 32, 8},                 // Erosion
    {0xefc8ef4d36102b34ULL, 0x1beeeb324a0f24eaULL, -7, 6, {1, 2, 1, 0, 0, 0}, 15.0 / 12, 40, 6}};             // Weirdness

// Starting lacunarity and persistence for a climate noise,
// lacuna_ini and persist_ini in cubiomes
__constant__ double LACUNARITY_START[13] = {1, .5, .25, 1. / 8, 1. / 16, 1. / 32, 1. / 64, 1. / 128,
                                            1. / 256, 1. / 512, 1. / 1024, 1. / 2048, 1. / 4096};
__constant__ double PERSISTENCE_START[10] = {0, 1, 2. / 3, 4. / 7, 8. / 15, 16. / 31, 32. / 63,
                                             64. / 127, 128. / 255, 256. / 511};

// The second Perlin noise multiplies the position by this
#define SECOND_SCALE (337.0 / 331.0)

// fade curve for Perlin noise
DEV double fade(double fraction) {
    return fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0);
}

/**
 * @brief Shuffles the permutation array for an octave and fills in the rest of the octave
 * 
 * @param octave The octave (Octave or OctaveHeader)
 * @param permutation The permutation array to shuffle into
 * @param random The random generator for this octave
 */
template <class OctaveType>
DEV void shuffleOctave(OctaveType *octave, uint8_t *permutation, XoroshiroState *random, double amplitude, double lacunarity) {
    octave->offsetX = xNextDouble(random) * 256.0;
    double offsetY = xNextDouble(random) * 256.0;
    octave->offsetZ = xNextDouble(random) * 256.0;

    // Start with 0, 1, 2 ... 255, we fill it in 8 bytes at a time
    uint64_t *words = (uint64_t *) permutation;
    for (int i = 0; i < 32; i++) {
        words[i] = 0x0706050403020100ULL + 0x0808080808080808ULL * (uint64_t) i;
    }

    for (int i = 0; i < 256; i++) {
        int j = xNextInt(random, 256 - i) + i;
        uint8_t temp = permutation[i];
        permutation[i] = permutation[j];
        permutation[j] = temp;
    }
    permutation[256] = permutation[0];

    double yFloor = floor(offsetY);
    double yFraction = offsetY - yFloor;
    octave->yLattice = (uint8_t) (int) yFloor;
    octave->yFraction = yFraction;
    octave->yFade = fade(yFraction);
    octave->amplitude = amplitude;
    octave->lacunarity = lacunarity;
}

/**
 * @brief Builds one octave of a climate noise
 * 
 * @param octave Where the octave goes (Octave or OctaveHeader)
 * @param permutation The permutation array to shuffle into
 * @param octaveIndex Index of the octave in the octave array
 * @param seedLow The low random number of the world seed
 * @param seedHigh The high random number of the world seed
 */
template <class OctaveType>
DEV void buildOctave(OctaveType *octave, uint8_t *permutation, int octaveIndex, uint64_t seedLow, uint64_t seedHigh) {
    // Find which climate value this octave belongs to
    int climateIndex = 0;
    while (climateIndex < 5 && octaveIndex >= CLIMATE_NOISE[climateIndex + 1].firstSlot) {
        climateIndex++;
    }
    const ClimateNoise &noise = CLIMATE_NOISE[climateIndex];

    int localIndex = octaveIndex - noise.firstSlot;
    int secondHalf = localIndex >= noise.slotCount / 2; // whether or not it's in the second Perlin noise
    int wantedOctave = localIndex - secondHalf * (noise.slotCount / 2);

    XoroshiroState random;
    random.low = seedLow ^ noise.saltLow;
    random.high = seedHigh ^ noise.saltHigh;
    uint64_t firstLow = xNextLong(&random);
    uint64_t firstHigh = xNextLong(&random);
    uint64_t secondLow = xNextLong(&random);
    uint64_t secondHigh = xNextLong(&random);

    uint64_t halfLow = firstLow;
    uint64_t halfHigh = firstHigh;
    if (secondHalf) {
        halfLow = secondLow;
        halfHigh = secondHigh;
    }

    double lacunarity = LACUNARITY_START[-noise.firstOctave];
    double persistence = PERSISTENCE_START[noise.amplitudeCount];
    int octavesSeen = 0;
    for (int i = 0; i < noise.amplitudeCount; i++, lacunarity *= 2.0, persistence *= 0.5) {
        if (noise.amplitudes[i] == 0) {
            continue;
        }
        if (octavesSeen == wantedOctave) {
            XoroshiroState octaveRandom;
            octaveRandom.low = halfLow ^ OCTAVE_SALTS[12 + noise.firstOctave + i][0];
            octaveRandom.high = halfHigh ^ OCTAVE_SALTS[12 + noise.firstOctave + i][1];
            shuffleOctave(octave, permutation, &octaveRandom, noise.amplitudes[i] * persistence, lacunarity);
            return;
        }
        octavesSeen++;
    }
}

// Dot product of the gradient the hash picks with the offset (indexedLerp in cubiomes)
DEV float gradientDot(uint8_t hash, float fractionX, float fractionY, float fractionZ) {
    switch (hash & 0xf) {
    case 0:
        return fractionX + fractionY;
    case 1:
        return -fractionX + fractionY;
    case 2:
        return fractionX - fractionY;
    case 3:
        return -fractionX - fractionY;
    case 4:
        return fractionX + fractionZ;
    case 5:
        return -fractionX + fractionZ;
    case 6:
        return fractionX - fractionZ;
    case 7:
        return -fractionX - fractionZ;
    case 8:
        return fractionY + fractionZ;
    case 9:
        return -fractionY + fractionZ;
    case 10:
        return fractionY - fractionZ;
    case 11:
        return -fractionY - fractionZ;
    case 12:
        return fractionX + fractionY;
    case 13:
        return -fractionY + fractionZ;
    case 14:
        return -fractionX + fractionY;
    default:
        return -fractionY - fractionZ;
    }
}

DEV float interpolate(float part, float start, float end) {
    return start + part * (end - start);
}

/**
 * @brief Samples a Perlin octave at y = 0, with floats for the gradient math
 * @param octave The octave (an Octave or an OctaveHeader)
 * @param permutation The shuffled permutation array of the octave
 * @param noiseX The x position in the noise
 * @param noiseZ The z position in the noise
 */
template <class OctaveType>
DEV float samplePerlinOctave(const OctaveType *octave, const uint8_t *permutation, double noiseX, double noiseZ) {
    float fadeY = (float) octave->yFade;
    uint8_t latticeY = octave->yLattice;
    noiseX += octave->offsetX;
    noiseZ += octave->offsetZ;
    double floorX = floor(noiseX);
    double floorZ = floor(noiseZ);
    noiseX -= floorX;
    noiseZ -= floorZ;

    uint8_t latticeX = (uint8_t) (int) floorX;
    uint8_t latticeZ = (uint8_t) (int) floorZ;
    float fadeX = (float) fade(noiseX);
    float fadeZ = (float) fade(noiseZ);
    float fractionX = (float) noiseX;
    float fractionY = (float) octave->yFraction;
    float fractionZ = (float) noiseZ;

    // hash the 8 corners of the cell
    uint8_t hashX0 = permutation[latticeX] + latticeY;
    uint8_t hashX1 = permutation[latticeX + 1] + latticeY;
    uint8_t hashX0Y0 = permutation[hashX0] + latticeZ;
    uint8_t hashX1Y0 = permutation[hashX1] + latticeZ;
    uint8_t hashX0Y1 = permutation[hashX0 + 1] + latticeZ;
    uint8_t hashX1Y1 = permutation[hashX1 + 1] + latticeZ;

    float corner1 = gradientDot(permutation[hashX0Y0], fractionX, fractionY, fractionZ);
    float corner2 = gradientDot(permutation[hashX1Y0], fractionX - 1, fractionY, fractionZ);
    float corner3 = gradientDot(permutation[hashX0Y1], fractionX, fractionY - 1, fractionZ);
    float corner4 = gradientDot(permutation[hashX1Y1], fractionX - 1, fractionY - 1, fractionZ);
    float corner5 = gradientDot(permutation[hashX0Y0 + 1], fractionX, fractionY, fractionZ - 1);
    float corner6 = gradientDot(permutation[hashX1Y0 + 1], fractionX - 1, fractionY, fractionZ - 1);
    float corner7 = gradientDot(permutation[hashX0Y1 + 1], fractionX, fractionY - 1, fractionZ - 1);
    float corner8 = gradientDot(permutation[hashX1Y1 + 1], fractionX - 1, fractionY - 1, fractionZ - 1);

    corner1 = interpolate(fadeX, corner1, corner2);
    corner3 = interpolate(fadeX, corner3, corner4);
    corner5 = interpolate(fadeX, corner5, corner6);
    corner7 = interpolate(fadeX, corner7, corner8);
    corner1 = interpolate(fadeY, corner1, corner3);
    corner5 = interpolate(fadeY, corner5, corner7);
    return interpolate(fadeZ, corner1, corner5);
}

/**
 * @brief Samples the double Perlin noise of a climate value at y = 0, from octaves that are already built
 * @param octaves The octave array
 * @param climateIndex Which climate value (see CLIMATE_NOISE)
 * @param noiseX The x position in the noise
 * @param noiseZ The z position in the noise
 * @return double The noise value
 */
DEV double sampleClimate(const Octave *octaves, int climateIndex, double noiseX, double noiseZ) {
    const ClimateNoise &noise = CLIMATE_NOISE[climateIndex];
    int halfCount = noise.slotCount / 2;
    double total = 0;

    for (int i = 0; i < halfCount; i++) {
        const Octave *octave = octaves + noise.firstSlot + i;
        double scale = octave->lacunarity;
        total += octave->amplitude * (double) samplePerlinOctave(octave, octave->permutation, noiseX * scale, noiseZ * scale);
    }

    for (int i = 0; i < halfCount; i++) {
        const Octave *octave = octaves + noise.firstSlot + halfCount + i;
        double scale = octave->lacunarity;
        total += octave->amplitude * (double) samplePerlinOctave(octave, octave->permutation, noiseX * SECOND_SCALE * scale, noiseZ * SECOND_SCALE * scale);
    }

    return total * noise.amplitude;
}

// Any low enough depth keeps the cave biomes out of reach and doesn't change the surface biome. genlut.c has to use the same value
#define FIXED_DEPTH (-15000)

/**
 * Gets the six climate values at a point (in 1:4 biome cells) and scales them like cubiomes does. They go into climate.
 */
DEV void climateAt(const Octave *octaves, int positionX, int positionZ, int64_t climate[6]) {
    // No coordinate warp here, the CPU check of the hits still does it. Skipping it lost about 1 in 60 of the high scoring seeds in testing
    float temperature = (float) sampleClimate(octaves, 1, positionX, positionZ);
    float humidity = (float) sampleClimate(octaves, 2, positionX, positionZ);
    float continentalness = (float) sampleClimate(octaves, 3, positionX, positionZ);
    float erosion = (float) sampleClimate(octaves, 4, positionX, positionZ);
    float weirdness = (float) sampleClimate(octaves, 5, positionX, positionZ);

    climate[0] = (int64_t) (10000.0f * temperature);
    climate[1] = (int64_t) (10000.0f * humidity);
    climate[2] = (int64_t) (10000.0f * continentalness);
    climate[3] = (int64_t) (10000.0f * erosion);
    climate[4] = FIXED_DEPTH;
    climate[5] = (int64_t) (10000.0f * weirdness);
}
