// Shuffles the permutation arrays in shared memory and saves the gradient hashes the sample grid needs
#pragma once

#define SHUFFLE_THREADS 96    // Threads in a block for the kernels that shuffle in shared memory
#define SHUFFLE_CHUNK (1 << 18) // The most seeds a build kernel gets in one go

// Byte i of a thread's permutation array. The threads of a block take turns in shared memory, a word at a time
#define TABLE_BYTE(table, i) (table)[((i) >> 2) * (SHUFFLE_THREADS * 4) + ((i) & 3)]

// Returns an octave offset (xNextDouble() * 256) without the doubles. The top 8 bits are the lattice cell and the rest
// is the fraction
DEV uint32_t octaveOffset(uint64_t randomLong) {
    return (uint32_t) (randomLong >> 32);
}

// The sample grid positions times the lacunarity, in fixed point [half][lacunarity exponent][step]. uploadGridPositions
// fills this in
__constant__ int32_t GRID_POSITIONS[2][11][9];

// The fraction part of a fixed point number, as a float
DEV float fractionOf(uint32_t noise) {
    return (float) (noise & 0xFFFFFFu) * (1.0f / 16777216.0f);
}

// fade curve for Perlin noise
DEV float fade(float fraction) {
    return fraction * fraction * fraction * (fraction * (fraction * 6.0f - 15.0f) + 10.0f);
}

// The rest of what a build kernel saves for a half (the offsets, the y fraction and the y fade)
struct HalfHeader {
    uint32_t offsetX;
    uint32_t offsetZ;
    float yFraction;
    float yFade;
};

// Xors three numbers in one instruction
DEV uint32_t xor3(uint32_t first, uint32_t second, uint32_t third) {
    uint32_t result;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(first), "r"(second), "r"(third));
    return result;
}

// Returns the high half of value * multiplier, plus add
DEV uint32_t multiplyHighAdd(uint32_t value, uint32_t multiplier, uint32_t add) {
    uint32_t result;
    asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(value), "r"(multiplier), "r"(add));
    return result;
}

// Multiplying by PIPE_ONE puts an add on the multiply-add pipe, since the compiler can't tell what it is (the ALU pipe is the
// busy pipe). The screening shuffles also leave out the carry from the low half of a draw into the high half, that almost never
// changes an array. shuffleShared builds the cascade's arrays and does the whole add
__constant__ uint32_t PIPE_ONE = 1;

/**
 * @brief Shuffles a thread's permutation array the way Minecraft does. If xNextInt might have had to draw again,
 * the shuffle gets redone the slow way
 * 
 * @param table The thread's array (see TABLE_BYTE)
 * @param tableWords table, as 32 bit words
 * @param random The octave's random generator (after the offsets)
 */
DEV void shuffleShared(uint8_t *table, uint32_t *tableWords, XoroshiroState *random) {
    XoroshiroState start = *random;
    for (int word = 0; word < 64; word++) {
        tableWords[word * SHUFFLE_THREADS] = 0x03020100u + 0x04040404u * (uint32_t) word;
    }

    // xNextLong written out on 32 bit halves
    uint32_t lowLow = (uint32_t) random->low;
    uint32_t lowHigh = (uint32_t) (random->low >> 32);
    uint32_t highLow = (uint32_t) random->high;
    uint32_t highHigh = (uint32_t) (random->high >> 32);
    // xNextInt draws again if the low word of a product is under the bound, we check for that at the end
    uint32_t lowestProduct = 0xFFFFFFFFu;
    uint8_t *wordBytes = table;
#pragma unroll 1
    for (int i = 0; i < 256; i += 4, wordBytes += SHUFFLE_THREADS * 4) {
#pragma unroll
        for (int part = 0; part < 4; part++) {
            uint32_t bound = 256 - (i + part);
            uint64_t sum = ((uint64_t) lowHigh << 32 | lowLow) + ((uint64_t) highHigh << 32 | highLow);
            uint32_t result = __funnelshift_l((uint32_t) (sum >> 32), (uint32_t) sum, 17) + lowLow;
            uint32_t xorLow = lowLow ^ highLow;
            uint32_t xorHigh = lowHigh ^ highHigh;
            uint32_t nextLowLow = xor3(__funnelshift_r(lowLow, lowHigh, 15), xorLow, xorLow << 21);
            uint32_t nextLowHigh = xor3(__funnelshift_r(lowHigh, lowLow, 15), xorHigh, __funnelshift_l(xorLow, xorHigh, 21));
            highLow = __funnelshift_l(xorHigh, xorLow, 28);
            highHigh = __funnelshift_l(xorLow, xorHigh, 28);
            lowLow = nextLowLow;
            lowHigh = nextLowHigh;

            lowestProduct = min(lowestProduct, result * bound);
            // The byte to swap with, xNextInt(random, bound) + i in the slow version
            uint32_t partner = multiplyHighAdd(result, bound, part);
            uint8_t *other = wordBytes + (int) (partner * (SHUFFLE_THREADS) - (partner & 3) * (SHUFFLE_THREADS - 1));
            uint8_t temp = wordBytes[part];
            wordBytes[part] = *other;
            *other = temp;
        }
    }
    random->low = (uint64_t) lowHigh << 32 | lowLow;
    random->high = (uint64_t) highHigh << 32 | highLow;

    if (lowestProduct < 256) {
        *random = start;
        for (int word = 0; word < 64; word++) {
            tableWords[word * SHUFFLE_THREADS] = 0x03020100u + 0x04040404u * (uint32_t) word;
        }
        for (int i = 0; i < 256; i++) {
            int j = xNextInt(random, 256 - i) + i;
            uint8_t temp = TABLE_BYTE(table, i);
            TABLE_BYTE(table, i) = TABLE_BYTE(table, j);
            TABLE_BYTE(table, j) = temp;
        }
    }
}

// Threads in a block for the kernels that shuffle in the shared tables
#define TABLE_THREADS 128
#define TABLE_BLOCKS 3 // blocks on an SM at once

// The number of 64 bit words a build kernel saves for a seed (the corner slices of both halves, then the two headers)
#define BUILT_WORDS(side) (2 * ((side) + 1) + 4)

// More multipliers the compiler can't see through, they move the shuffle's bound and address math onto the multiply-add pipe
__constant__ int32_t PIPE_MINUS_ONE = -1;
__constant__ uint32_t PIPE_QUARTER = 1u << 30;

// A thread's array is a column of words with 4 entries in a word, and the words are TABLE_THREADS apart. That way the lanes of
// a warp hit their own banks, and the shuffle can store a word of finished entries in a single write
DEV uint8_t *tableStart(uint32_t *tableWords) {
    return (uint8_t *) (tableWords + threadIdx.x);
}

// Returns where entry position is, starting from tableStart
DEV uint32_t tableOffset(uint32_t position) {
    return (position >> 2) * (TABLE_THREADS * 4) + (position & 3);
}

// Fills all of the block's arrays with 0 to 255 (all of the threads in the block have to call this)
DEV void resetTables(uint32_t *tableWords) {
    uint4 *quads = (uint4 *) tableWords;
    for (int quad = threadIdx.x; quad < 16 * TABLE_THREADS; quad += TABLE_THREADS) {
        uint32_t entry = 0x03020100u + 0x04040404u * (uint32_t) (quad / (TABLE_THREADS / 4)); // a row of words is TABLE_THREADS / 4 quads
        quads[quad] = make_uint4(entry, entry, entry, entry);
    }
}

// Returns entry position of a thread's array (position wraps around)
DEV uint32_t tableEntry(const uint8_t *table, uint32_t position) {
    return table[tableOffset(position & 255)];
}

// A xoroshiro generator in 32 bit halves, like shuffleShared keeps it
struct SplitRandom {
    uint32_t lowLow;
    uint32_t lowHigh;
    uint32_t highLow;
    uint32_t highHigh;
};

// Returns the low 32 bits of xNextLong (that's all xNextInt needs)
DEV uint32_t nextLowWord(SplitRandom &random) {
    uint32_t sumLow = random.lowLow * PIPE_ONE + random.highLow;   // plain adds on the multiply-add pipe, with no carry
    uint32_t sumHigh = random.lowHigh * PIPE_ONE + random.highHigh;
    uint32_t result = __funnelshift_l(sumHigh, sumLow, 17) + random.lowLow; // the low word of rotateLeft(sum, 17)
    uint32_t xorLow = random.lowLow ^ random.highLow;
    uint32_t xorHigh = random.lowHigh ^ random.highHigh;
    uint32_t nextLowLow = xor3(__funnelshift_r(random.lowLow, random.lowHigh, 15), xorLow, xorLow << 21);
    uint32_t nextLowHigh = xor3(__funnelshift_r(random.lowHigh, random.lowLow, 15), xorHigh, __funnelshift_l(xorLow, xorHigh, 21));
    random.highLow = __funnelshift_l(xorHigh, xorLow, 28);
    random.highHigh = __funnelshift_l(xorLow, xorHigh, 28);
    random.lowLow = nextLowLow;
    random.lowHigh = nextLowHigh;
    return result;
}

/**
 * @brief Shuffles a thread's array the way Minecraft does, like shuffleShared, but it skips xNextInt's redraw (that hardly
 * ever changes the result)
 * 
 * @param table The thread's table (see tableStart), it has to hold 0 to 255 already
 * @param random The octave's random generator (after the offsets)
 */
DEV void shuffleTable(uint8_t *table, XoroshiroState random) {
    SplitRandom split;
    split.lowLow = (uint32_t) random.low;
    split.lowHigh = (uint32_t) (random.low >> 32);
    split.highLow = (uint32_t) random.high;
    split.highHigh = (uint32_t) (random.high >> 32);

    // Entry i is finished when step i swaps it, and nothing reads it again in this shuffle. A step just writes the entry it swaps
    // with, and the finished entries get stored a word at a time
    uint8_t *row = table; // entry i
#pragma unroll 1
    for (int i = 0; i < 256; i += 16, row += 16 * TABLE_THREADS) {
#pragma unroll
        for (int word = 0; word < 4; word++) {
            uint32_t finished = 0;
#pragma unroll
            for (int part = 4 * word; part < 4 * word + 4; part++) {
                uint32_t bound = (uint32_t) (i * PIPE_MINUS_ONE + (256 - part));
                uint32_t result = nextLowWord(split);
                // The entry to swap with, as a distance from entry i (tableOffset works on the distance since i is a multiple of 4)
                uint32_t distance = multiplyHighAdd(result, bound, part);
                uint32_t quarter = __umulhi(distance, PIPE_QUARTER); // distance >> 2
                uint8_t *other = row + (distance + quarter * (TABLE_THREADS * 4 - 4));
                uint8_t current = row[(part >> 2) * (TABLE_THREADS * 4) + (part & 3)];
                uint8_t swapped = *other;
                *other = current;
                finished = (uint32_t) swapped * (1u << (8 * (part & 3))) + finished;
            }
            *(uint32_t *) (row + word * (TABLE_THREADS * 4)) = finished;
        }
    }
}

/**
 * @brief Looks up the gradient hashes of all the lattice corners a block of side x side cells touches. slices[z] gets the
 * corners at that z, a byte for each x (the low 4 bits for y, the high 4 for y + 1)
 * 
 * @param table The thread's shuffled table
 * @param startX The first lattice cell in x
 * @param latticeY The lattice cell in y
 * @param startZ The first lattice cell in z
 * @param side Cells on a side of the block
 * @param slices Gets side + 1 slices
 */
DEV void cornerSlices(const uint8_t *table, uint32_t startX, uint32_t latticeY, uint32_t startZ, int side, uint64_t *slices) {
    uint32_t hashXY[7][2];
#pragma unroll
    for (int x = 0; x <= side; x++) {
        uint32_t hashX = tableEntry(table, startX + x) + latticeY;
        hashXY[x][0] = tableEntry(table, hashX) + startZ;
        hashXY[x][1] = tableEntry(table, hashX + 1) + startZ;
    }

#pragma unroll
    for (int z = 0; z <= side; z++) {
        uint64_t slice = 0;
#pragma unroll
        for (int x = 0; x <= side; x++) {
            slice |= (uint64_t) (tableEntry(table, hashXY[x][0] + z) & 15) << (8 * x);
            slice |= (uint64_t) (tableEntry(table, hashXY[x][1] + z) & 15) << (8 * x + 4);
        }
        slices[z] = slice;
    }
}

/**
 * @brief Shuffles the octave for one half of a climate value in the thread's table and gets its corner slices
 * 
 * @param tableWords The block's shared arrays
 * @param table The thread's array (see tableStart)
 * @param halfLow The low random number of the half
 * @param halfHigh The high one
 * @param saltIndex Where the octave's salt is in OCTAVE_SALTS
 * @param half 0 for the first Perlin noise, 1 for the second
 * @param lacunarityExponent The octave's lacunarity is 2 to the minus this
 * @param side Cells on a side of the block the grid can touch
 * @param slices Gets the corner slices (see cornerSlices)
 * @param header Gets the offsets and the y fraction and fade
 */
DEV void buildHalf(uint32_t *tableWords, uint8_t *table, uint64_t halfLow, uint64_t halfHigh, int saltIndex, int half, int lacunarityExponent, int side,
                   uint64_t *slices, HalfHeader *header) {
    XoroshiroState octaveRandom;
    octaveRandom.low = halfLow ^ OCTAVE_SALTS[saltIndex][0];
    octaveRandom.high = halfHigh ^ OCTAVE_SALTS[saltIndex][1];
    header->offsetX = octaveOffset(xNextLong(&octaveRandom));
    uint64_t yRandom = xNextLong(&octaveRandom);
    header->offsetZ = octaveOffset(xNextLong(&octaveRandom));

    // Everyone has to be done with the tables before they get reset
    __syncthreads();
    resetTables(tableWords);
    __syncthreads();
    shuffleTable(table, octaveRandom);

    header->yFraction = fractionOf(octaveOffset(yRandom));
    header->yFade = fade(header->yFraction);
    uint32_t startX = (header->offsetX + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][0]) >> 24;
    uint32_t startZ = (header->offsetZ + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][0]) >> 24;
    cornerSlices(table, startX, (uint32_t) (yRandom >> 56), startZ, side, slices);
}

// Threads in a block for the kernels that shuffle the halves together (a thread gets two of the array slots)
#define FUSED_THREADS (TABLE_THREADS / 2)

/**
 * @brief Shuffles the arrays for the halves of a climate value in a single loop, like shuffleTable does for a single
 * array. The two xoroshiro chains are independent, the scheduler can overlap them
 *
 * @param table0 The first half's array
 * @param table1 The second half's array
 * @param random0 The first half's octave generator (after the offsets)
 * @param random1 The second half's
 */
DEV void shuffleTwoTables(uint8_t *table0, uint8_t *table1, XoroshiroState random0, XoroshiroState random1) {
    SplitRandom split0, split1;
    split0.lowLow = (uint32_t) random0.low;
    split0.lowHigh = (uint32_t) (random0.low >> 32);
    split0.highLow = (uint32_t) random0.high;
    split0.highHigh = (uint32_t) (random0.high >> 32);
    split1.lowLow = (uint32_t) random1.low;
    split1.lowHigh = (uint32_t) (random1.low >> 32);
    split1.highLow = (uint32_t) random1.high;
    split1.highHigh = (uint32_t) (random1.high >> 32);
    uint8_t *row0 = table0, *row1 = table1;
#pragma unroll 1
    for (int i = 0; i < 256; i += 16, row0 += 16 * TABLE_THREADS, row1 += 16 * TABLE_THREADS) {
#pragma unroll
        for (int word = 0; word < 4; word++) {
            uint32_t finished0 = 0, finished1 = 0;
#pragma unroll
            for (int part = 4 * word; part < 4 * word + 4; part++) {
                uint32_t bound = (uint32_t) (i * PIPE_MINUS_ONE + (256 - part));
                uint32_t result0 = nextLowWord(split0);
                uint32_t result1 = nextLowWord(split1);
                uint32_t distance0 = multiplyHighAdd(result0, bound, part);
                uint32_t distance1 = multiplyHighAdd(result1, bound, part);
                uint8_t *other0 = row0 + (distance0 + __umulhi(distance0, PIPE_QUARTER) * (TABLE_THREADS * 4 - 4));
                uint8_t *other1 = row1 + (distance1 + __umulhi(distance1, PIPE_QUARTER) * (TABLE_THREADS * 4 - 4));
                uint8_t current0 = row0[(part >> 2) * (TABLE_THREADS * 4) + (part & 3)];
                uint8_t current1 = row1[(part >> 2) * (TABLE_THREADS * 4) + (part & 3)];
                uint8_t swapped0 = *other0;
                uint8_t swapped1 = *other1;
                *other0 = current0;
                *other1 = current1;
                finished0 = (uint32_t) swapped0 * (1u << (8 * (part & 3))) + finished0;
                finished1 = (uint32_t) swapped1 * (1u << (8 * (part & 3))) + finished1;
            }
            *(uint32_t *) (row0 + word * (TABLE_THREADS * 4)) = finished0;
            *(uint32_t *) (row1 + word * (TABLE_THREADS * 4)) = finished1;
        }
    }
}

/**
 * @brief Builds the octave for the halves of a climate value at once, like buildHalf does for a single half. All of the
 * block's threads have to call this (it resets all of the array slots)
 *
 * @param tableWords The block's shared arrays
 * @param halfLow The low random numbers of the halves
 * @param halfHigh The high ones
 * @param saltIndex Where the octave's salt is in OCTAVE_SALTS
 * @param lacunarityExponent The octave's lacunarity is 2 to the minus this
 * @param side Cells on a side of the block the grid can touch
 * @param slices0 Gets the first half's corner slices
 * @param slices1 Gets the second half's
 * @param header0 Gets the first half's offsets, y fraction and y fade
 * @param header1 Gets the second half's
 */
DEV void buildTwoHalves(uint32_t *tableWords, const uint64_t *halfLow, const uint64_t *halfHigh, int saltIndex, int lacunarityExponent, int side,
                        uint64_t *slices0, uint64_t *slices1, HalfHeader *header0, HalfHeader *header1) {
    XoroshiroState random[2];
    uint64_t yRandom[2];
    HalfHeader *headers[2] = {header0, header1};
#pragma unroll
    for (int half = 0; half < 2; half++) {
        random[half].low = halfLow[half] ^ OCTAVE_SALTS[saltIndex][0];
        random[half].high = halfHigh[half] ^ OCTAVE_SALTS[saltIndex][1];
        headers[half]->offsetX = octaveOffset(xNextLong(&random[half]));
        yRandom[half] = xNextLong(&random[half]);
        headers[half]->offsetZ = octaveOffset(xNextLong(&random[half]));
    }

    // Reset all of the arrays, then shuffle
    __syncthreads();
    uint4 *quads = (uint4 *) tableWords;
    for (int quad = threadIdx.x; quad < 16 * TABLE_THREADS; quad += FUSED_THREADS) {
        uint32_t entry = 0x03020100u + 0x04040404u * (uint32_t) (quad / (TABLE_THREADS / 4));
        quads[quad] = make_uint4(entry, entry, entry, entry);
    }
    __syncthreads();
    uint8_t *table0 = (uint8_t *) (tableWords + threadIdx.x), *table1 = (uint8_t *) (tableWords + threadIdx.x + FUSED_THREADS);
    shuffleTwoTables(table0, table1, random[0], random[1]);

    // Look up the corner slices for the halves
#pragma unroll
    for (int half = 0; half < 2; half++) {
        headers[half]->yFraction = fractionOf(octaveOffset(yRandom[half]));
        headers[half]->yFade = fade(headers[half]->yFraction);
        uint32_t startX = (headers[half]->offsetX + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][0]) >> 24;
        uint32_t startZ = (headers[half]->offsetZ + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][0]) >> 24;
        cornerSlices(half ? table1 : table0, startX, (uint32_t) (yRandom[half] >> 56), startZ, side, half ? slices1 : slices0);
    }
}

// Gets the random numbers the two halves of a climate value start from
DEV void climateHalves(uint64_t seed, uint64_t saltLow, uint64_t saltHigh, uint64_t halfLow[2], uint64_t halfHigh[2]) {
    XoroshiroState random;
    xSetSeed(&random, seed);
    uint64_t seedLow = xNextLong(&random);
    uint64_t seedHigh = xNextLong(&random);
    XoroshiroState climateRandom;
    climateRandom.low = seedLow ^ saltLow;
    climateRandom.high = seedHigh ^ saltHigh;
    halfLow[0] = xNextLong(&climateRandom);
    halfHigh[0] = xNextLong(&climateRandom);
    halfLow[1] = xNextLong(&climateRandom);
    halfHigh[1] = xNextLong(&climateRandom);
}

// Saves a half's corner slices and header for a seed (built is the seed's first word, and the next ones are stride apart)
DEV void saveBuiltHalf(uint64_t *built, int side, int half, const uint64_t *slices, const HalfHeader &header, size_t stride = SHUFFLE_CHUNK) {
    for (int z = 0; z <= side; z++) {
        built[(size_t) (half * (side + 1) + z) * stride] = slices[z];
    }
    const int headerWord = 2 * (side + 1) + half * 2;
    built[(size_t) headerWord * stride] = header.offsetX | (uint64_t) header.offsetZ << 32;
    built[(size_t) (headerWord + 1) * stride] = __float_as_uint(header.yFraction) | (uint64_t) __float_as_uint(header.yFade) << 32;
}

// Reads back what saveBuiltHalf saved
DEV void readBuiltHalf(const uint64_t *built, int side, int half, uint64_t *slices, HalfHeader *header, size_t stride = SHUFFLE_CHUNK) {
    for (int z = 0; z <= side; z++) {
        slices[z] = built[(size_t) (half * (side + 1) + z) * stride];
    }
    const int headerWord = 2 * (side + 1) + half * 2;
    uint64_t offsets = built[(size_t) headerWord * stride];
    uint64_t fractions = built[(size_t) (headerWord + 1) * stride];
    header->offsetX = (uint32_t) offsets;
    header->offsetZ = (uint32_t) (offsets >> 32);
    header->yFraction = __uint_as_float((uint32_t) fractions);
    header->yFade = __uint_as_float((uint32_t) (fractions >> 32));
}

// The rest of samplePerlinOctave, with two corner slices shifted down to the cell. nearZ is the slice at the cell's z and farZ
// the next one
DEV float sampleCorners(uint64_t nearZ, uint64_t farZ, float fractionX, float fractionY, float fractionZ, float fadeX, float fadeY, float fadeZ) {
    uint32_t nearHashes = (uint32_t) nearZ;
    uint32_t farHashes = (uint32_t) farZ;
    float corner1 = gradientDot(nearHashes, fractionX, fractionY, fractionZ);
    float corner2 = gradientDot(nearHashes >> 8, fractionX - 1, fractionY, fractionZ);
    float corner3 = gradientDot(nearHashes >> 4, fractionX, fractionY - 1, fractionZ);
    float corner4 = gradientDot(nearHashes >> 12, fractionX - 1, fractionY - 1, fractionZ);
    float corner5 = gradientDot(farHashes, fractionX, fractionY, fractionZ - 1);
    float corner6 = gradientDot(farHashes >> 8, fractionX - 1, fractionY, fractionZ - 1);
    float corner7 = gradientDot(farHashes >> 4, fractionX, fractionY - 1, fractionZ - 1);
    float corner8 = gradientDot(farHashes >> 12, fractionX - 1, fractionY - 1, fractionZ - 1);

    corner1 = interpolate(fadeX, corner1, corner2);
    corner3 = interpolate(fadeX, corner3, corner4);
    corner5 = interpolate(fadeX, corner5, corner6);
    corner7 = interpolate(fadeX, corner7, corner8);
    corner1 = interpolate(fadeY, corner1, corner3);
    corner5 = interpolate(fadeY, corner5, corner7);
    return interpolate(fadeZ, corner1, corner5);
}

// The four numbers a row of samples needs from a lattice cell, collapseRowCell works them out
struct RowCell {
    float nearSlope;
    float farSlope;
    float nearConstant;
    float farConstant;
};

// The gradient a hash picks, as x, y and z
DEV void gradientVector(uint32_t hash, float *gradientX, float *gradientY, float *gradientZ) {
    hash &= 15;

    bool firstIsX = (0x50FFu >> hash) & 1;
    bool secondIsY = (0x500Fu >> hash) & 1;
    float first = (0xEAAAu >> hash) & 1 ? -1.0f : 1.0f;
    float second = (0x8CCCu >> hash) & 1 ? -1.0f : 1.0f;
    *gradientX = firstIsX ? first : 0.0f;
    *gradientY = (firstIsX ? 0.0f : first) + (secondIsY ? second : 0.0f);
    *gradientZ = secondIsY ? 0.0f : second;
}

// The gradients for the 16 hashes (the temperature kernel keeps a copy in shared memory)
struct GradientTable {
    float2 xy[16];
    float z[16];
};

// Fills in a GradientTable, the threads of the block split up the work
DEV void fillGradientTable(GradientTable *table) {
    for (int hash = threadIdx.x; hash < 16; hash += blockDim.x) {
        float gradientX, gradientY, gradientZ;
        gradientVector((uint32_t) hash, &gradientX, &gradientY, &gradientZ);
        table->xy[hash] = make_float2(gradientX, gradientY);
        table->z[hash] = gradientZ;
    }
}

// Boils a lattice cell down to the four numbers a row of samples needs (see RowCell). nearZ has the hashes of the corners at
// the row's z, farZ the ones at z + 1, and cell is the cell's x in the slices
DEV void collapseRowCell(uint32_t nearZ, uint32_t farZ, int cell, float fractionY, float fadeY, float fractionZ, float fadeZ, RowCell *result,
                         const GradientTable *gradients) {
    float slope[2], constant[2];

#pragma unroll
    for (int corner = 0; corner < 2; corner++) { // the x corners
        float zSlope[2], zConstant[2];
#pragma unroll
        for (int back = 0; back < 2; back++) { // the z corners
            uint32_t slice = back ? farZ : nearZ;
            uint32_t lowHash = (slice >> (8 * (cell + corner))) & 15;
            uint32_t highHash = (slice >> (8 * (cell + corner) + 4)) & 15;
            float2 lowXY = gradients->xy[lowHash];
            float2 highXY = gradients->xy[highHash];
            float lowZ = gradients->z[lowHash];
            float highZ = gradients->z[highHash];

            float alongZ = back == 0 ? fractionZ : fractionZ - 1.0f;
            zSlope[back] = interpolate(fadeY, lowXY.x, highXY.x);
            float low = fmaf(lowZ, alongZ, lowXY.y * fractionY);
            float high = fmaf(highZ, alongZ, highXY.y * (fractionY - 1.0f));
            zConstant[back] = interpolate(fadeY, low, high);
        }
        slope[corner] = interpolate(fadeZ, zSlope[0], zSlope[1]);
        constant[corner] = interpolate(fadeZ, zConstant[0], zConstant[1]);
    }

    result->nearSlope = slope[0];
    result->farSlope = slope[1];
    result->nearConstant = constant[0];
    result->farConstant = constant[1];
}

// The fade, fraction and cell for the 9 columns (or rows) of the sample grid, worked out ahead of time
struct GridSteps {
    float fade[9];
    float fraction[9];
    int cell[9];
};

/**
 * @brief Fills in the steps for a half of an octave, like samplePerlinOctave does (in fixed point)
 * 
 * @param steps Where they go
 * @param half 0 for the first Perlin noise, 1 for the second
 * @param lacunarityExponent The octave's lacunarity is 2 to the minus this
 * @param offset offsetX for columns, offsetZ for rows
 */
DEV void fillGridSteps(GridSteps *steps, int half, int lacunarityExponent, uint32_t offset) {
    uint32_t startCell = (offset + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][0]) >> 24;
#pragma unroll
    for (int step = 0; step < 9; step++) {
        uint32_t noise = offset + (uint32_t) GRID_POSITIONS[half][lacunarityExponent][step];
        steps->cell[step] = (uint8_t) ((noise >> 24) - startCell);
        float fraction = fractionOf(noise);
        steps->fade[step] = fade(fraction);
        steps->fraction[step] = fraction;
    }
}

/**
 * @brief Builds one octave of a climate noise, shuffled in shared memory
 * 
 * @param octave Gets the header
 * @param table The thread's permutation array (see TABLE_BYTE)
 * @param tableWords table, as 32 bit words
 * @param octaveIndex Index in the octave array
 * @param seedLow The low random number of the world seed
 * @param seedHigh The high random number of the world seed
 */
DEV void buildOctave(OctaveHeader *octave, uint8_t *table, uint32_t *tableWords, int octaveIndex, uint64_t seedLow, uint64_t seedHigh) {
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

    uint64_t halfLow = firstLow, halfHigh = firstHigh;
    if (secondHalf) {
        halfLow = secondLow;
        halfHigh = secondHigh;
    }

    double persistence = PERSISTENCE_START[noise.amplitudeCount];
    int octavesSeen = 0;
    for (int i = 0; i < noise.amplitudeCount; i++, persistence *= 0.5) {
        if (noise.amplitudes[i] == 0) {
            continue;
        }
        if (octavesSeen == wantedOctave) {
            XoroshiroState octaveRandom;
            octaveRandom.low = halfLow ^ OCTAVE_SALTS[12 + noise.firstOctave + i][0];
            octaveRandom.high = halfHigh ^ OCTAVE_SALTS[12 + noise.firstOctave + i][1];
            octave->offsetX = octaveOffset(xNextLong(&octaveRandom));
            uint64_t yRandom = xNextLong(&octaveRandom);
            octave->offsetZ = octaveOffset(xNextLong(&octaveRandom));
            shuffleShared(table, tableWords, &octaveRandom);

            octave->yLattice = (uint8_t) (yRandom >> 56);
            octave->yFraction = fractionOf(octaveOffset(yRandom));
            octave->yFade = fade(octave->yFraction);
            octave->amplitude = (float) (noise.amplitudes[i] * persistence);
            octave->shift = (uint8_t) (16 - noise.firstOctave - i); // the lacunarity is 2 to the (firstOctave + i)
            octave->secondHalf = (uint8_t) secondHalf;
            return;
        }
        octavesSeen++;
    }
}
