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

// A Perlin noise octave (the offsets are in fixed point)
struct Octave {
    uint8_t permutation[257];
    uint8_t yLattice;
    uint8_t shift;      // how far to shift the position for this octave
    uint8_t secondHalf; // 1 if it's in the second Perlin noise
    uint32_t offsetX;
    uint32_t offsetZ;
    float amplitude;
    float yFraction;
    float yFade;
};

// An Octave without the permutation array. The probe and build kernels use this and keep the permutations seperate
struct OctaveHeader {
    uint32_t offsetX;
    uint32_t offsetZ;
    float amplitude;
    float yFraction;
    float yFade;
    uint8_t yLattice;
    uint8_t shift;
    uint8_t secondHalf;
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

// Starting persistence for a climate noise, persist_ini in cubiomes
__constant__ double PERSISTENCE_START[10] = {0, 1, 2. / 3, 4. / 7, 8. / 15, 16. / 31, 32. / 63,
                                             64. / 127, 128. / 255, 256. / 511};

// The second Perlin noise multiplies the position by this
#define SECOND_SCALE (337.0 / 331.0)

// Dot product of the gradient the hash picks with the offset (indexedLerp in cubiomes), with bit masks and not a switch
DEV float gradientDot(uint32_t hash, float fractionX, float fractionY, float fractionZ) {
    hash &= 15;
    float first = (0x50FFu >> hash) & 1 ? fractionX : fractionY;
    float second = (0x500Fu >> hash) & 1 ? fractionY : fractionZ;
    first = __int_as_float(__float_as_int(first) ^ (int) (((0xEAAAu >> hash) & 1) << 31));
    second = __int_as_float(__float_as_int(second) ^ (int) (((0x8CCCu >> hash) & 1) << 31));
    return first + second;
}

DEV float interpolate(float part, float start, float end) {
    return start + part * (end - start);
}

// The amplitudes of the double Perlin noises in CLIMATE_NOISE, as floats
__constant__ float CLIMATE_AMPLITUDES[6] = {(float) (15.0 / 12), (float) (15.0 / 12), (float) (10.0 / 9),
                                            (float) (45.0 / 30), (float) (25.0 / 18), (float) (15.0 / 12)};

// SECOND_SCALE times 2^40, for the fixed point positions
#define SECOND_SCALE_FIXED 1119442352147LL

/**
 * @brief Samples a Perlin octave at y = 0, with floats for the gradient math
 * @param octave The octave
 * @param noiseX The x position in the noise in fixed point, with the offset added
 * @param noiseZ The z position
 */
DEV float samplePerlinOctave(const Octave *octave, uint32_t noiseX, uint32_t noiseZ) {
    const uint8_t *permutation = octave->permutation;
    float fadeY = octave->yFade;
    uint8_t latticeY = octave->yLattice;
    uint8_t latticeX = (uint8_t) (noiseX >> 24);
    uint8_t latticeZ = (uint8_t) (noiseZ >> 24);
    float fractionX = (float) (noiseX & 0xFFFFFFu) * (1.0f / 16777216.0f);
    float fractionZ = (float) (noiseZ & 0xFFFFFFu) * (1.0f / 16777216.0f);
    float fadeX = fractionX * fractionX * fractionX * (fractionX * (fractionX * 6.0f - 15.0f) + 10.0f);
    float fadeZ = fractionZ * fractionZ * fractionZ * (fractionZ * (fractionZ * 6.0f - 15.0f) + 10.0f);
    float fractionY = octave->yFraction;

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

// A position (in 1:4 biome cells) for the first and second Perlin noise, in fixed point
struct NoisePosition {
    int64_t first;
    int64_t second;
};

// Returns the NoisePosition for a position
DEV NoisePosition toNoisePosition(int position) {
    NoisePosition fixed;
    fixed.first = (int64_t) position << 40;
    fixed.second = (int64_t) position * SECOND_SCALE_FIXED;
    return fixed;
}

/**
 * @brief Samples the double Perlin noise of a climate value at y = 0, from octaves that are already built
 * @param octaves The octave array
 * @param climateIndex Which climate value (see CLIMATE_NOISE)
 * @param positionX The x position from toNoisePosition
 * @param positionZ The z position from toNoisePosition
 * @return float The noise value
 */
DEV float sampleClimate(const Octave *octaves, int climateIndex, NoisePosition positionX, NoisePosition positionZ) {
    const ClimateNoise &noise = CLIMATE_NOISE[climateIndex];
    int slotCount = noise.slotCount;
    float total = 0;

    for (int i = 0; i < slotCount; i++) {
        const Octave *octave = octaves + noise.firstSlot + i;
        int shift = octave->shift;
        int64_t scaledX = octave->secondHalf ? positionX.second : positionX.first;
        int64_t scaledZ = octave->secondHalf ? positionZ.second : positionZ.first;
        uint32_t noiseX = (uint32_t) (scaledX >> shift) + octave->offsetX;
        uint32_t noiseZ = (uint32_t) (scaledZ >> shift) + octave->offsetZ;
        total += octave->amplitude * samplePerlinOctave(octave, noiseX, noiseZ);
    }
    return total * CLIMATE_AMPLITUDES[climateIndex];
}

// Any low enough depth keeps the cave biomes out of reach and doesn't change the surface biome. genlut.c has to use the same value
#define FIXED_DEPTH (-15000)

/**
 * Gets the six climate values at a point (in 1:4 biome cells) and scales them like cubiomes does. They go into climate.
 */
DEV void climateAt(const Octave *octaves, int positionX, int positionZ, int64_t climate[6]) {
    // No coordinate warp here, the CPU check of the hits still does it. Skipping it lost about 1 in 60 of the high scoring seeds in testing
    NoisePosition noiseX = toNoisePosition(positionX);
    NoisePosition noiseZ = toNoisePosition(positionZ);
    float temperature = sampleClimate(octaves, 1, noiseX, noiseZ);
    float humidity = sampleClimate(octaves, 2, noiseX, noiseZ);
    float continentalness = sampleClimate(octaves, 3, noiseX, noiseZ);
    float erosion = sampleClimate(octaves, 4, noiseX, noiseZ);
    float weirdness = sampleClimate(octaves, 5, noiseX, noiseZ);

    // These are nowhere near 2^31, an int is plenty and it's a lot faster
    climate[0] = (int) (10000.0f * temperature);
    climate[1] = (int) (10000.0f * humidity);
    climate[2] = (int) (10000.0f * continentalness);
    climate[3] = (int) (10000.0f * erosion);
    climate[4] = FIXED_DEPTH;
    climate[5] = (int) (10000.0f * weirdness);
}
