# JUNO
JUNO (Just Use Noise Once) is a GPU seed finder for Minecraft 26.3 - it looks for the seeds with the most even mix of biomes around the world origin. This is release v1.3.

## Quick start
```sh
make          # build it (Linux or WSL2, with gcc and the CUDA toolkit, see Building it)
make test     # optional, checks the CPU side of the build. It should say PASS
make run      # start searching in the background
make log      # watch for new records (Ctrl+C closes the log, the search keeps going)
make stop     # stop searching, make run carries on from there next time
```

`make run` gives you your terminal back right away:

```
$ make run
JUNO v1.3
There's no checkpoint yet, so this starts at index 0
Picked a random custom seed and saved it to ../results/custom_seed.txt: h1OZIrYad5z9qGHL
The custom seed moves the stream by 14890994237487958828
JUNO is running in the background. make log shows the new records, make status shows how it's going and make stop stops it.
```

`make log` then shows each new record as it comes in, a hit with a better score than any hit before it:

```
$ make log
Showing new records (Ctrl+C closes the log, JUNO keeps running)
[00:38:06] New record (SENTS and ARBITRATIONS)! seed -9041054982481344052  SENTS 0.906393542  ARBITRATIONS 82.144499766
[00:38:08] New record (SENTS and ARBITRATIONS)! seed -2949509553309681244  SENTS 0.907009186  ARBITRATIONS 82.248762707
[00:38:10] New record (SENTS and ARBITRATIONS)! seed 244007842027566613  SENTS 0.912024061  ARBITRATIONS 83.174149635
```

The first time, records come in within seconds on my machine, and after that they get rarer. Records are counted against every hit you already have though, so later on `make log` starts with a "Best so far" line and a new record can take hours. So if you just want to know it's actually doing something, run `make status` after half a minute and check the progress goes up and there's a speed. Every hit (SENTS 0.905 and up) also goes into `results/gpu_hits.jsonl`, about 65 a minute here, and `make top N=10` lists the best ones.

`make run` only checks the range before it says it's running, the GPU gets set up after that in the background. So if the GPU part fails (not enough GPU memory, the wrong ARCH for your card), the scanner stops, `make status` shows `Scanner: not running`, and the reason is in `results/gpu_scan.log`.

## What it does
- Scores seeds with SENTS (Scaled ENTropy Score), basically how evenly the surface biomes are spread over the $4096 \times 4096$ block square centred on the origin
- Minecraft 26.3, all 52 above-ground overworld biomes
- Billions of seeds a second (a quick check at the start throws out most of them before the expensive part).
- Hits get scored again on the CPU with cubiomes before they're saved, and you get their ARBITRATIONS score too.
- You decide what SENTS or ARBITRATIONS score a hit needs to get saved
- Everyone gets their own stream of seeds. The first run picks a random custom seed and keeps it, so people scanning at the same time don't redo each other's work.
- Runs in the background until you stop it, restarts itself if it crashes, and checkpoints every 30 seconds, so you can stop it whenever and pick up where you left off
- `make log` shows new records as they come in, `make status` shows how it's doing, `make score` scores any seed on the CPU and `make top` goes through your hits.

## The score
For a set of $N$ biomes, where $p_i$ is the fraction of the area that biome $i$ covers ($i = 1, \dots, N$), the score of that seed (SENTS) is given by this piecewise function:

```math
\mathrm{SENTS} = \begin{cases}
0 & \text{if any of the } N \text{ biomes are missing} \\
-\dfrac{1}{\log N} \displaystyle\sum_{i=1}^{N} p_i \log p_i & \text{otherwise}
\end{cases}
```

We use $N = 52$ (the above-ground overworld biomes of Minecraft 26.3). A score of 1 would mean every one of the 52 biomes covers exactly $1/52$ of the area (which is actually impossible since $52 \nmid 4096^2$)

The square is read as a $1024 \times 1024$ grid of 1:4 biome cells, above ground at height $y = 256$. At sea level, high terrain gives you the cave biome at that height and not the surface biome above it, which throws off the areas and can even hide a biome completely.

The biomes come from cubiomes, a C version of Minecraft's world generator. The parts JUNO uses live in `vendor/cubiomes`, extended to 26.3 (see `vendor/NOTICE.md`).

### ARBITRATIONS
This is the score arb4096_crunch_final uses. JUNO saves it too, which makes it easy to compare hits with theirs. It reads a $1025 \times 1025$ grid, the $1024 \times 1024$ grid with one extra row and column at the edges. With $q_i$ the fraction of that grid that biome $i$ covers, and $H = -\sum_{i=1}^{52} q_i \ln q_i$ the entropy of those fractions,

```math
\mathrm{ARBITRATIONS} = 100 \left( \frac{H}{\ln 52} \right)^2
```

Unlike SENTS it doesn't drop to 0 if a biome is missing.

## How it works
Scoring a seed properly means reading all $1024 \times 1024$ cells, which takes a few seconds on one CPU core. That's way too slow for billions of seeds. So JUNO throws out as many as it can with quick checks first, and only does the full scan for the few that look good.

The scanner goes through a stream of indexes (an index gets turned into a seed with a multiply and an XOR, see Custom seeds). A batch of indexes goes through these steps:

1. The gate. It looks at a few octave offsets of the seed (humidity, erosion and weirdness), on the GPU (or on CPU threads with `--cpu-gate`). Only 3.5% of the indexes get through (2.5% with `--high-value`).
2. The temperature kernel builds the temperature noise and samples a $9 \times 9$ grid. It checks how evenly the samples spread over the temperature bands, then checks the temperature score.
3. The humidity kernel adds humidity at those points and checks the cell score.
4. The probe kernel adds continentalness and erosion, then checks the probe score.
5. The build kernel makes the octaves for whatever seeds are left, and the cascade kernel counts their biomes on a grid that gets finer level by level until it has all $1024 \times 1024$ cells. A level doesn't redo cells from the levels before it, and a seed that makes it all the way costs about one full scan. The rank score and the proximity gate throw seeds out along the way (levels 6 and 7 also skip a cell when the biome around it is obvious).
6. The CPU check. Hits get scored again with cubiomes before they're saved.

Step 6 is there because the GPU cuts corners to go faster (floats, no coordinate warp). Anything the GPU scores at 0.903 or above goes to the CPU check, and the CPU gets the final say.

The gate and the scores in steps 2 to 5 were fitted to good seeds, and between them and skipping the warp a good seed gets thrown out every now and then. It's fast, but it does miss some. Don't assume a range it has scanned has every good seed in it.

Getting all of this to work took alot of trial and error. [FILTERS.md](FILTERS.md) goes through the filters one at a time (where the numbers came from, and some of what I tried that didn't work), and compares JUNO with arb4096_crunch_final.

## Building it
You'll need Linux on an x86-64 CPU, gcc, make and the CUDA toolkit. WSL2 on Windows works too, that's what I run it on. I've only tested it with CUDA 12.0 (Ubuntu's `nvidia-cuda-toolkit` package) and an RTX 4080 SUPER. If something breaks on your setup, open an issue.

```sh
make
```

That compiles the cubiomes files, the CPU gate and the scanner in the `cuda` folder. The first time it also builds and runs `genlut`, which makes the lookup table (`lut263.bin` and `lut263.h`). `make test` then scores a seed whose score is known on the CPU and says PASS or FAIL. That checks cubiomes and the CPU side of the build, but not the GPU part (`make run` and `make status` do that). `make clean` deletes everything it built, the lookup table included. `make help` lists all of the commands.

By default it builds for an RTX 40 series card (`sm_89`). For another card use `ARCH` with your card's compute capability, e.g. `make ARCH=sm_86` for the 30 series. It remembers the card (even after `make clean`), so you only have to do that once. You'll want about 6 GB of free GPU memory, or about 1.5 GB with `--streams 1` (it's slower though).

Build it on the computer you're going to run it on, the CPU gate (for `--cpu-gate` and `--cpu-assist`) gets compiled with `-march=native`. If your CPU doesn't have AVX-512 the gate falls back to a slower version (when the scanner starts it writes which one it's using to `results/gpu_scan.log`). That one keeps a few more seeds and is several times slower per thread.

## Running
```sh
make run          # start in the background, carrying on from the checkpoint (the first time, from index 0)
make log          # watch for new records
make status       # what's running, how far it got, the speed and the best hit
make stop         # stop everything
```

`make run` checks the range and saves the checkpoint, then starts the watchdog in the background and gives you your terminal back. The watchdog keeps `cuda/supervise.sh` going, and the supervisor keeps the scanner going, so if the scanner crashes it starts again from the checkpoint. It keeps going until you run `make stop`, which lets the batches that are already on the GPU finish, waits for the hit checks and saves the checkpoint (that usually takes a few seconds).

`make log` shows new records by default. To see every new hit with at least a certain score instead, give it a minimum (or both, then a hit needs both):

```sh
make log MIN_SENTS=0.915
make log MIN_ARBITRATIONS=84
```

Ctrl+C only closes the log (make says `Interrupt` when you do that, which is normal). You can open and close it as often as you want while JUNO runs, in any terminal. It only shows what happens while it's open: new hits, what the supervisor says (starts, stops, giving up), and a few of the scanner's messages (like "Couldn't make the GPU buffers" or "Stopped"). Some errors, like a CUDA error, only go to `results/gpu_scan.log`.

You can also scan a set part of the stream with a start index, and a count if you want the range to end:

```sh
make run START=5000000000000 COUNT=1000000000000
```

START and COUNT have to be written out in full (1000000000000, not 1e12). Giving a START starts a new range and saves over the checkpoint. When a range with a COUNT is done, the supervisor starts the range right after it, with the same COUNT. To change anything while JUNO is running, `make stop` first.

Only one scanner can run on a computer at a time. A second one refuses to start, and so does `--prepare` (they check the lock file `/tmp/juno-scan.lock`), because on WSL a second scanner doesn't get turned down when it asks for more GPU memory, and two of them at once can freeze Windows.

A hit gets saved when its SENTS score is at least 0.905. You can raise that, or ask for an ARBITRATIONS score too, with `OPTIONS`:

```sh
make run OPTIONS="--min-arbitrations 84"
```

The OPTIONS only count for that `make run` (the watchdog keeps using them until `make stop`), they don't get saved anywhere. So give them again the next time, after a restart too.

| Option | What it does |
| --- | --- |
| `--min-sents <score>` | The SENTS score a hit needs to get saved (0.905 if you leave it out) |
| `--min-arbitrations <score>` | The ARBITRATIONS score a hit needs to get saved |
| `--custom-seed <text>` | Scan the stream picked by this text (see Custom seeds). With `make run`, use `CUSTOM_SEED=<text>` |
| `--plain-stream` | Scan the plain stream, without a custom seed |
| `--high-value` | Tighter GPU filters and a narrower gate that go after the best hits (ARBITRATIONS 85 and up). See Looking for the best seeds |
| `--all-hits` | The wider filters again. This is what you get anyway, it's there to turn `--high-value` back off |
| `--gate-rate <percent>` | The percent of the indexes the gate lets through (3.5 if you leave it out, 2.5 with `--high-value`, 10 with `--high-value --cpu-gate`) |
| `--cpu-gate` | Run the gate on the CPU threads, the way v1.1 did. The GPU does it about ten times faster, so this is mostly there for comparing |
| `--cpu-assist` | The CPU threads gate part of every batch, so the GPU gate has less to do. About 10% faster with `--high-value` (next to nothing with the default settings), but it keeps your CPU busy |
| `--gate-threads <n>` | Threads for the CPU gate with `--cpu-gate` or `--cpu-assist` (all but one or two of your CPU threads with `--cpu-gate`, all but four with `--cpu-assist`, if you leave it out) |
| `--streams <n>` | Batches the GPU works on at once (5 if you leave it out). Fewer of them needs less GPU memory, for cards with less to spare |

The SENTS minimum always applies (0.905 unless you raise it), so a hit also needs all 52 biomes even if you only ask for an ARBITRATIONS score. The GPU filters are tuned for hits with $\mathrm{SENTS} \ge 0.905$, and the scanner won't take anything lower. For a hit with all 52 biomes $\mathrm{ARBITRATIONS} \approx 100 \cdot \mathrm{SENTS}^2$, which puts the lowest ARBITRATIONS minimum at $100 \cdot 0.905^2 = 81.9025$ (so `--min-arbitrations 81.9` gets turned down).

### Looking for the best seeds
By default JUNO saves every hit from SENTS 0.905 up and lets 3.5% of the indexes through the gate. `--high-value` goes after the top of the tail instead: every filter gets tighter and the gate goes down to 2.5%, so almost nothing under ARBITRATIONS 84.25 makes it through.

```sh
make run OPTIONS="--high-value"
```

v1.3 raised its probe cutoff from 11 to 12, which loses 1 of the 67 seeds over ARBITRATIONS 86 that I know of (all 4 over 87 stay). Up to v1.2 it used a 7% gate. At 2.5% the scanner gets through about 2.4 times as many indexes a second, and a bit over half of the seeds over 87 that 7% lets through are still inside 2.5%, so it should find about 1.3 times as many of them an hour. That comes from a model of where the hits over 84 sit in the gate (anywhere from 1.05 to 1.6 times fits the data), since a hit over 87 only turns up about once every 30 hours, which is too rare to count directly.

A higher gate rate lets through more of the good seeds for each index, but the GPU has more to work through, so fewer indexes go by a second. 3.5% was the best for the default settings when I tested it. For `--high-value` anything from 2% to 3% came out about the same, so it uses 2.5%.

You can run the scanner yourself from the `cuda` folder too, in the foreground, with the options above: `./scan [options] run` or `./scan [options] <start index> [count]`. Run it from inside that folder though, it looks for `lut263.bin` and `../results` relative to where it's run. It only prints a few lines when it starts, and a summary when it stops (Ctrl+C stops it the same way `make stop` does, and a second Ctrl+C stops it right away). `make log` and `make status` still work while it runs, but nothing restarts it if it crashes. There's also `--prepare`, which only checks the range and saves the checkpoint (that's the first thing `make run` does).

### Checking on it
`make status` prints something like this:

```
Watchdog:    running
Supervisor:  running
Scanner:     running
Saved seed:  h1OZIrYad5z9qGHL (the custom seed new ranges use)
Progress:    128.4 billion indexes done, starting from index 0 (this range goes until you stop it)
             (the checkpoint was saved 17 seconds ago)
Speed:       about 10.4 billion indexes a second
Hits:        9 saved in results/gpu_hits.jsonl
Best hit:    seed 244007842027566613  SENTS 0.912024061  ARBITRATIONS 83.174150
```

### Checking seeds
```sh
make score SEED=3549742502234867244           # SENTS and ARBITRATIONS of a seed, on the CPU
make top N=10                                 # the 10 best hits in results/gpu_hits.jsonl
make top N=10 BY=arbitrations                 # the same, sorted by ARBITRATIONS
```

`make score` uses the same CPU check the scanner does before it saves a hit, so each seed takes a few seconds. You can give it more than one seed with `SEED="<seed> <seed>"`. `make top` only lists a seed once, even if it got saved more than once.

### Custom seeds
Index $j$ of the stream is the seed $\gamma j \oplus \sigma$, where $\gamma = \mathtt{0x9E3779B97F4A7C15}$, $\sigma = \mathtt{0x6A09E667F3BCC909}$, $\oplus$ is XOR and the math wraps around at $2^{64}$. That means two people who scan one range get identical seeds, which is a waste. A custom seed gives you your own stream.

You don't have to pick one. The first time you start a range without `CUSTOM_SEED`, the scanner picks a random custom seed (16 letters and digits), saves it to `results/custom_seed.txt` and prints it. After that every new range uses the saved one. You can also give your own:

```sh
make run CUSTOM_SEED='bryCE219!'
```

The scanner hashes the text (printable ASCII) into a 64-bit number $k$, using FNV-1a and then $\mathrm{mix}$ (the output mix of splitmix64), and index $j$ becomes the seed $\gamma (k + j) \oplus \sigma$. For `bryCE219!` that's $k = 13456412924314603523$. $k$ gets saved in the checkpoint. After that `make run` stays in your stream without the text. If you already have a checkpoint for a different stream, add `START=0` to start a range in the new one.

Careful with two things here. A new START without CUSTOM_SEED uses the saved seed from `results/custom_seed.txt`, not the one in the checkpoint, so give CUSTOM_SEED again when you start a new range in your own stream. And there's only one checkpoint, so switching to another stream loses your place in the old one. If you want to go back to it later, write down the checkpoint first (`results/gpu_progress.txt`, the numbers in `make status` are rounded). It's `juno-gpu-v1 <k> <start> <count> <done>`, so the START to carry on from is start + done, and if that range had a COUNT, the COUNT that's left is count - done.

To scan the plain stream (to repeat someone else's range, say), give the range and add `OPTIONS=--plain-stream`. If you delete the `results` folder, the next range gets a new random custom seed, so keep a copy of `custom_seed.txt` if you want to know which part of which stream you've scanned.

Two people with different custom seeds who scan $L$ indexes each overlap with a chance of approximately $2L / 2^{64}$. For a year of scanning on my machine that's a few percent (about 4%, or 7% with `--high-value`), and even then it'd just be part of the ranges.

Up to v1.2 index $j$ was the seed $\mathrm{mix}(\gamma j)$. Minecraft starts a seed's random numbers from $\mathrm{mix}(s \oplus \sigma)$ and $\mathrm{mix}((s \oplus \sigma) + \gamma)$, and for $s = \gamma j \oplus \sigma$ those are $\mathrm{mix}(\gamma j)$ and $\mathrm{mix}(\gamma (j + 1))$, so the next index reuses one of them. That takes the gate from three mixes an index down to one. The catch is that an index gives a different seed than it did in v1.2. A v1.2 checkpoint still works, `make run` carries on from the same index with the new seeds. The seeds v1.2 went through and the ones after them are two separate random picks, so a seed after the switch was already scanned with a chance of only $D / 2^{64}$, where $D$ is how many indexes v1.2 got through (about 1 in 5,000 after 3.4 quadrillion).

### Where the results go
The scanner saves into the `results` folder at the top of the project, and makes the folder if it isn't there.
- `results/gpu_progress.txt` is the checkpoint. It gets saved every 30 seconds and when the scanner stops, and it's one line: `juno-gpu-v1 <k> <start> <count> <done>`. Starting a new range saves over it. Without a COUNT the count is the biggest one allowed (9223372036850581503), which would take about 70 years at 4 billion a second.
- `results/gpu_hits.jsonl` gets the hits as their CPU check finishes (see The hit file below).
- `results/custom_seed.txt` has the custom seed your new ranges use (see Custom seeds).
- `results/gpu_scan.log` has what the scanner and the supervisor print (errors included), and `results/watchdog.log` what the watchdog does.
- `results/gpu_speed.txt` is the speed, for `make status`.

The gate runs on the GPU. With `--cpu-gate` it uses all your CPU threads but one or two, and with `--cpu-assist` all but four.

### If something goes wrong
If the scanner stops 3 times in a row without saving any progress (usually a GPU problem, like running out of GPU memory or a CUDA error), the supervisor gives up. `make status` then shows the watchdog running but the supervisor and the scanner not. The reason is in `results/gpu_scan.log`, right above the line where the supervisor gives up (`make log` might not show the reason, and only shows anything if it was open at the time). Fix the problem, then `make stop` and `make run`.

To stop just the supervisor, `touch results/STOP_GPU` (a scan that's already running keeps going). The watchdog leaves the supervisor alone while that file is there, and `touch results/STOP_WATCHDOG` stops the watchdog. `make stop` does all of that for you.

IMPORTANT NOTE: JUNO doesn't start when your computer boots, so run `make run` again after a restart.

## The hit file
Each hit is one JSON line:

```json
{"seed": 3549742502234867244, "sents": 0.910265368, "arbitrations": 82.850095608, "missing": 0, "mc": "26.3", "side": 4096, "y": 256, "src": "gpu", "finder": "JUNO v1.1", "cpu_verified": true, "found": "2026-09-15T09:08:40Z"}
```

`seed` is the number you'd type into Minecraft (it can be negative). `sents` is the SENTS score from the CPU check and `arbitrations` is the ARBITRATIONS score (defined [above](#arbitrations)). `cpu_verified` says whether or not the GPU and the CPU agreed on the score, and `found` is the date and time the hit got saved (in UTC). The rest (`missing`, `mc`, `side`, `y`, `src`, `finder`) don't change from hit to hit within a version. `missing` is how many biomes the GPU didn't find, so it's always 0 on a saved hit.

## Known issues
- If the scanner gets killed instead of stopped (`kill -9`, a crash or a power cut), the indexes after the last checkpoint get scanned again next time, so hits from that stretch can show up twice in the hits file. Nothing gets skipped though: the checkpoint never moves past a hit that's still in its CPU check.
- `make stop` and `make status` only look at the copy of JUNO they're run in (they go by the folder a process runs in). The lock only knows about JUNO scanners, so `make run` and the supervisor also won't start a scanner while any other program called `scan` is running.
- `make score`, `make top` and `make test` run the same `scan` program. While one of them is going, `make run` says JUNO is already running, `make status` shows the scanner as running, the supervisor won't restart a crashed scanner, and `make stop` stops it (a long `make score` just gets cut off). They're usually done in a few seconds.
- The scanner uses the first GPU. You can pick a different one with `CUDA_VISIBLE_DEVICES`.
- Without AVX-512 the CPU gate is a lot slower (see Building it).
- The CPU checks one hit for each of its threads at a time. With a fast GPU and a slow CPU the GPU ends up waiting for it, and `--min-sents` or `--high-value` gives it fewer hits to check.

## Performance
On my Ryzen 9 7950X3D and RTX 4080 SUPER the scanner gets through about 11.9 billion stream indexes per second (19.3 billion with `--high-value`), with about 415 million a second making it past the gate. `--cpu-assist` adds another 10% with `--high-value`, and next to nothing with the default settings (there the CPU threads are busy checking hits anyway). The settings were tuned on that machine (the `ARCH` default in the `Makefile`, and the thread and stream counts in `cuda/scan.cu`), so your numbers will be different.

## Credits
- Cubitect, for cubiomes, which all of the biome generation here is built on.
- arb4096: the idea behind the offset gate comes from the crunch filter in arb4096_crunch_final, and the temperature band edges and the idea of weighting the bands by how many biomes they hold come from their HIP biome scanner.
- The 26.3 biome tree comes from the cubiomes fork that shipped with the HIP scanner.

## License
MIT, see `LICENSE`. cubiomes keeps its own MIT licence in `vendor/cubiomes/LICENSE`.
