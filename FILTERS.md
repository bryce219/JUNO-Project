# JUNO's filters

Everything JUNO does to avoid scoring a seed properly, in the order it happens. A SENTS score (the Scaled ENTropy Score, defined in the README) reads all $1024 \times 1024$ biome cells and takes a few seconds on a CPU core, and almost no seed is worth that. A seed has to pass each check to get to the next one. I've tried to note where the numbers came from where I could.

## 1. The CPU gate

This one runs before the GPU ever sees a seed, and it doesn't build any noise.

Some background first. The climate values in Minecraft (humidity, erosion, weirdness and the rest) are made from two Perlin noises, and a Perlin noise is a stack of octaves. When an octave gets created it picks a random offset for x, y and z, and the gate just looks at the y offsets. Let $c$ be humidity, erosion or weirdness, $n \in \{1, 2\}$ one of the two Perlin noises of $c$, and $o \in \{1, 2\}$ one of the two lowest octaves in that noise (12 octaves in total). If $f_{c,n,o}$ is the fraction part of that octave's y offset and $a_{c,n,o}$ is the weight for it, a seed gets through when

```math
\sum_{c} \; \sum_{n=1}^{2} \; \sum_{o=1}^{2} a_{c,n,o} \left| f_{c,n,o} - \tfrac{1}{2} \right| \le 0.62184
```

The y offset matters because the climate noise is sampled at height $y = 0$, which means $f_{c,n,o}$ decides how the octave blends its gradients. I can't fully explain why good seeds mostly have it near $\tfrac{1}{2}$, but the data was pretty clear about it. I stuck to the lowest octaves since they have the biggest features (the ones that decide where the large biome regions end up). Working out an offset is a few random draws, which is the whole reason this is cheap.

The weights $a_{c,n,o}$ came from a logistic regression, 9,585 good seeds against 300,000 random ones (out of 4 million, the rest were held back to check the fit). The good seeds were all the hits an earlier version of JUNO found in 1.3 trillion seeds, back before this gate existed: all 52 biomes, with a CPU-checked SENTS between 0.8976 and 0.9211. The threshold of 0.62184 is simply what lets 1% of indexes through. It does throw out some good seeds, but the search finds good seeds several times faster with the gate on. When it first went in, the scanner went from under 10 million seeds a second (all of them on the GPU) to over 500 million indexes a second, with the GPU only seeing the 1% that passed. Everything since then has been alot of smaller changes on both sides.

One more detail. The AVX-512 version stops adding erosion and weirdness to a seed when the humidity part (the 4 terms where $c$ is humidity) goes over 0.5. Most indexes (around 70%) get dropped right there, and it only loses something like 1 passing seed in 20,000.

## 2. The temperature kernel

The first thing on the GPU. It builds the lowest octave of the two temperature Perlin noises and samples them at 81 points, a $9 \times 9$ grid with a point every 512 blocks from $-2048$ to $2048$. A sample lands in one of 5 temperature bands (the edges are $-0.45, -0.15, 0.2, 0.55$, which are the bands Minecraft itself uses to pick biomes). With $p_b$ the share of samples in band $b$, $w_b$ the weight of band $b$ and $W = \sum_{b=1}^{5} w_b = 478$, the band evenness is

```math
E_T = \frac{1}{\ln W} \sum_{b=1}^{5} p_b \ln \frac{w_b}{p_b}
```

Each band's weight is the sum of the weights of its 5 cells (5 bands of 5 cells, so 25 cells in all). A cell's weight is the number of the 52 biomes whose climate box in cubiomes reaches that cell (dappled forest has no box there, so it counts in all of them). Weighting the bands like this was their HIP scanner's idea. I did try weights computed straight from the biome tree, and they did worse.

There's a free cut here. The humidity kernel later checks this kind of evenness over 25 cells, and merging cells into bands can never make the evenness go down, meaning the cell evenness is always at most $E_T$. So a seed with $E_T < 0.9425$ could never pass the humidity kernel anyway, and dropping it now costs nothing. (The kernel actually goes a bit further and wants $E_T \ge 0.968$, because the lowest $E_T$ I found among the few thousand good seeds that made it to the cascade was about 0.972.)

Then there's the temperature score, which basically guesses whether the humidity kernel is going to keep the seed anyway. It's a logistic model on the band shares, $E_T$, the mean and spread of the samples, how many samples sit near a band edge, and a 16-bin histogram. Cut at 0.4127 it kept pretty much all (99.9%) of the good seeds the humidity kernel would have kept, at least when I fit it.

## 3. The humidity kernel

Adds the lowest octave of the two humidity Perlin noises at the same 81 points. The samples then go into 25 cells, 5 temperature bands times 5 humidity bands (edges $-0.35, -0.1, 0.1, 0.3$, again Minecraft's own).

Two cuts:
- the cell evenness, $E_C$, which is the formula for $E_T$ with the shares and weights of the 25 cells swapped in for the bands. It needs $E_C \ge 0.9425$. It's pretty loose, the cell score is what really filters here
- cell score $\ge 0.4642$, a fitted model on the cell shares, $E_C$ and the kind of temperature/humidity features the temperature score uses. At this cutoff it keeps all 465 known seeds with $\mathrm{SENTS} \ge 0.915$.

## 4. The probe kernel

Continentalness and erosion, at those 81 points again, using the two lowest octaves of their two Perlin noises. The probe score is the cell score's features plus the mean, spread, min, max and a 16-bin histogram of continentalness and of erosion, and it has to be at least 8.5187. Out of fold (with the cell score at an older cutoff) it kept about 93% of the good seeds between 0.905 and 0.910 and 98% of the ones at 0.910 and up. (I tried weirdness here too. It didn't help.)

## 5. The cascade

This is where the actual biomes get read. The cascade looks at a seed on finer and finer grids: level $\ell$ uses one cell out of every $2^{8-\ell}$ in each direction, starting with an $8 \times 8$ grid at level 1 and ending with all $1024 \times 1024$ cells at level 8. A level never redoes a cell from an earlier level. So a seed that makes it all the way through only costs one full scan. (An earlier version started with 16 cells. Starting at 64 was faster.)

Between levels the seed gets checked, and what gets checked depends on the level.

**Levels 1 to 5** use a rank score, a logistic model on the biome shares so far, the number of biomes found, the evenness, and the number of biomes with a single sample. Level 1's cutoff (0.0513) came out of a search for the fastest set of cutoffs. For levels 2 to 5 the cutoffs are 10, 16, 19 and 20. Out of the ~12,800 hits I had at the time, those kept every seed at $\mathrm{SENTS} \ge 0.910$.

**After level 3** comes the proximity gate. For the 12 rarest biomes, some sample's climate has to be within 2000 (cubiomes climate units) of that biome's box in the biome tree, otherwise the biome probably isn't going to show up at all and SENTS would be 0. A few notes on the numbers:
- level 3 is where the seed has 1,024 samples. I also tried a second gate at level 2, and it either didn't speed anything up or it lost a good seed
- 2000 is a margin for climates that fall in between the samples, and with it the gate didn't lose a single seed in testing
- 12 biomes and not 52, because checking all 52 cost about as much time as it saved

**Levels 6 and 7.** Let $m$ be the number of biomes found so far and $E$ the evenness of the samples so far (their entropy divided by $\ln 52$). The seed needs $m + E \ge 51.902$ after level 6, and $m + E \ge 52.902$ after level 7. So after level 6 it can still be missing one biome if $E \ge 0.902$, and after level 7 it needs all 52 biomes and $E \ge 0.902$. The 0.902 comes from the traces I looked at, where the evenness at these levels stayed within about 0.001 of the final score. That leaves some room under 0.905.

Levels 6 and 7 also skip cells. If the 4 points around a cell on the $128 \times 128$ grid all have biome $X$, and none of the biomes that are still missing have a climate anywhere close to $X$, the cell gets counted as $X$ without computing it. In testing this found the seeds the no-skipping version found, nothing lost.

**Level 8** is the full grid. A seed that gets there with a GPU score of 0.903 or more goes to the CPU check. The 0.002 of slack is there because the GPU uses floats and skips the coordinate warp; skipping the warp only moves the score by a tiny bit (less than 0.0004). The margin only covers the score though. Skipping the warp can also push a biome that covers a cell or two off the grid, and then the GPU gives the seed a 0. That cost 1 of 60 high-scoring seeds when I tested it.

## 6. The CPU check

The hit gets scored again with cubiomes (doubles, coordinate warp included) and saved if it clears the minimum scores.

## Shortcuts that don't change the answer

**Fixed depth.** At height $y = 256$ the depth stayed somewhere between about $-18000$ and $-4400$ when I measured it. All of the surface biome leaves have the same depth box, which means any depth that low gives you the surface biome you'd get with the real depth. JUNO just uses $-15000$ and skips the depth spline.

**Lookup table.** With the depth fixed you can cut the other five climate axes at all of the leaf edges in the biome tree. That gives $11 \times 12 \times 18 \times 15 \times 29 = 1{,}033{,}560$ cells, and inside a cell the biome doesn't change. The GPU looks the biome up in that table and never walks the tree.

**Less noise.** The first three kernels only build the octaves mentioned above (i.e. not the whole noise), and the GPU never builds the shift noise at all.

## Compared with previous work (arb4096_crunch_final)

Their search does one seed at a time on CPU threads:
1. 4 octave offsets with fixed cutoffs
2. temperature evenness $\ge 0.984$
3. a reduced pre-scan of all 5 climates
4. a $17 \times 17$ scan
5. a $65 \times 65$ scan
6. the full $1025 \times 1025$ scan

Steps 4 to 6 each start over from nothing and run every cell through cubiomes' depth spline and biome tree (steps 5 and 6 with the warp too). JUNO fits its filters to good seeds, does everything after the gate on the GPU, and doesn't compute a cell twice. On my machine JUNO finds about 130 seeds an hour at $\mathrm{SENTS} \ge 0.910$. For their program on all 32 threads I estimate around 1.6 an hour, so roughly 80 times less - but that's an estimate, not something I measured directly.

Credit where its due: the idea for the CPU gate comes from their crunch filter, and the temperature band edges and the idea of weighting the bands are from their HIP biome scanner.
