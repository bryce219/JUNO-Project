// Biome data for the proximity gate and for skipping cells in the cascade kernel
#pragma once

// If the samples don't get close to a rare biome's climate box, that biome probably won't show up when the grid gets finer. The proximity gate checks for this.
#define PROX_BIOMES 12 // how many biomes the proximity gate checks

// The slots (in SCORED_BIOMES) of those biomes, rarest first
__constant__ int PROXIMITY_ORDER[PROX_BIOMES] = {40, 11, 36, 26, 48, 25, 34, 47, 27, 32, 2, 17};

// Climate boxes for these biomes (temperature, humidity, continentalness, erosion, weirdness), a box covers
// all the leaves the biome has in the tree
__constant__ int BIOME_BOX_MIN[PROX_BIOMES][5] = {
  {   5500, -10000,  -1900, -10000,   -500},
  { -10000, -10000, -12000, -10000, -10000},
  { -10000, -10000,  -1900, -10000,   -500},
  {   5500,   1000,  -1900, -10000, -10000},
  {   2000, -10000,  -1100,   5500, -10000},
  {   5500, -10000,  -1900, -10000, -10000},
  { -10000, -10000,  -1900,   4500, -10000},
  {   2000, -10000,  -1900, -10000,  -9333},
  {   5500, -10000, -10500, -10000, -10000},
  { -10000, -10000, -10500, -10000, -10000},
  {   5500, -10000,  -1900, -10000, -10000},
  { -10000, -10000,  -1900,  -2225, -10000}};

__constant__ int BIOME_BOX_MAX[PROX_BIOMES][5] = {
  {  10000,  -1000,  10000,    500,  10000},
  {  10000,  10000, -10500,  10000,  10000},
  {  -4500,  -3500,  10000,  10000,  10000},
  {  10000,  10000,  10000,    500,  10000},
  {  10000,  10000,  10000,  10000,  10000},
  {  10000,   1000,  10000,    500,  10000},
  {  -1500,  -1000,  10000,   5500,  10000},
  {   5500,  10000,  10000,  -3750,   9333},
  {  10000,  10000,  -1900,  10000,  10000},
  {  -4500,  10000,  -4550,  10000,  10000},
  {  10000,  10000,  10000,  10000,  10000},
  {  -4500,  10000,  -1100,  10000,   2666}};

// Biomes (as bits) with a climate that's very different from the biome in a slot. Levels 6 and 7 use these to skip cells
__constant__ unsigned long long FAR_BIOMES[BIOME_COUNT] = {
  0x000605110e120f04ULL,   0x00000001c0008800ULL,   0x000e7c7fe07e8f69ULL,
  0x0006fd01cf018804ULL,   0x0008001bc0008800ULL,   0x0009831bdf80e804ULL,
  0x0006fd01cf018804ULL,   0x00026001c0008800ULL,   0x000787aa5f8ce805ULL,
  0x0007e3abdf8ce805ULL,   0x000583abdf8ce805ULL,   0x000ffffe37ff77ffULL,
  0x00004001c0008800ULL,   0x0008005fe0328f20ULL,   0x0008205fe0328f20ULL,
  0x000fffff0fff77feULL,   0x00010085c0408848ULL,   0x0005c3abdf8ce805ULL,
  0x0008011bce128f04ULL,   0x000a111fcf928f04ULL,   0x000d83bbdf8ce805ULL,
  0x000bb31fdf80e804ULL,   0x000efd1bcf018804ULL,   0x000c0255e03a8f20ULL,
  0x000d02d5e07a8f68ULL,   0x000f7cffe07e8f69ULL,   0x000f7cffe07e8f69ULL,
  0x000e7c7fa07e8769ULL,   0x000e0455a0320f20ULL,   0x000787005f806804ULL,
  0x000fffffa7ff77feULL,   0x000ffffe5fff76feULL,   0x000ffffe5ffff6ffULL,
  0x00042b71ce7eef34ULL,   0x0007ff41dfa9e804ULL,   0x00064b71ce7eef34ULL,
  0x0005abebdffce835ULL,   0x0008211bce128f04ULL,   0x000bd31fdf80e804ULL,
  0x0006fd11c7138f00ULL,   0x000f7effe07e8f69ULL,   0x000a315fe1b28f20ULL,
  0x00010185fe40894dULL,   0x0009019fce40884cULL,   0x000503c5ce68884cULL,
  0x000b43b7ce60caccULL,   0x000121cdce429accULL,   0x000900d5e0728f68ULL,
  0x000efd55e7338f20ULL,   0x000523cdfe688bcdULL,   0x000b119fffd28f4dULL,
  0x0005ab61dffce834ULL,
};
