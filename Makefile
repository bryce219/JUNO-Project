# Makefile for JUNO. Type make to build the scanner, or make help for the other commands
# ARCH is the graphics card to build for. It gets saved to cuda/arch.txt so a plain make uses the same one next time.
ARCH ?= $(shell cat cuda/arch.txt 2>/dev/null || echo sm_89)
# cubiomes folder, the cubiomes objects we build, and the scanner headers
CUB = vendor/cubiomes
CUBIOMES = cuda/cubiomes_noise.o cuda/cubiomes_biomes.o cuda/cubiomes_layers.o cuda/cubiomes_biomenoise.o
HEADERS = $(wildcard cuda/*.cuh cuda/*.inc cuda/*.h)

# CUSTOM_SEED goes to the scanner through the environment so any text (spaces, quotes, $ etc) stays in one piece
# make really wants to eat dollar signs otherwise
override CUSTOM_SEED := $(value CUSTOM_SEED)
export CUSTOM_SEED

.PHONY: all help run log status stop score top test clean FORCE

all: cuda/scan cuda/lut263.bin

help:
	@echo "make					Build the scanner (add ARCH=sm_86 etc for another card)"
	@echo "make test				Check the build by scoring a seed with a known score (CPU only)"
	@echo "make run				Start searching in the background, it carries on from the checkpoint"
	@echo "         START=<index>			...start a new range from this index instead"
	@echo "         COUNT=<count>			...and end that range after this many indexes (needs START)"
	@echo "         CUSTOM_SEED=<text>		...in your own stream (otherwise it uses the one saved in results/custom_seed.txt)"
	@echo "         OPTIONS=\"...\"			...with scanner options (--min-sents, --min-arbitrations, --plain-stream)"
	@echo "make log				Watch for new records as they come in (Ctrl+C just closes the log)"
	@echo "         MIN_SENTS=<score>		...show every new hit with at least this SENTS score instead"
	@echo "         MIN_ARBITRATIONS=<score>	...or at least this ARBITRATIONS score"
	@echo "make status				What's running, how far it got, the speed and the best hit"
	@echo "make stop				Stops everything (the scanner saves the checkpoint first)"
	@echo "make score SEED=<seed>			Score seeds on the CPU (SEED=\"<seed> <seed> ...\" for more than one)"
	@echo "make top N=<number>			List the best hits in results/gpu_hits.jsonl (add BY=arbitrations)"
	@echo "make clean				Delete everything make built (it still remembers ARCH)"

# The parts of cubiomes we use
cuda/cubiomes_%.o: $(CUB)/%.c $(wildcard $(CUB)/*.h)
	gcc -O3 -march=native -fwrapv -fPIC -c -o $@ $<

# genlut makes the lookup table
cuda/genlut: cuda/genlut.c $(CUBIOMES)
	gcc -O2 -I$(CUB) -o $@ cuda/genlut.c $(CUBIOMES) -lm

# Run genlut if there's no lookup table yet (it writes lut263.h too)
cuda/lut263.bin: | cuda/genlut
	cd cuda && ./genlut

# CPU gate (-march=native turns on AVX-512 if your CPU has it)
cuda/hostgate.o: cuda/hostgate.c cuda/hostgate.h
	gcc -O3 -march=native -I$(CUB) -c -o $@ cuda/hostgate.c

# Rewrite arch.txt if ARCH changed (a new ARCH rebuilds the scanner)
cuda/arch.txt: FORCE
	@echo '$(ARCH)' | cmp -s - $@ || echo '$(ARCH)' > $@

cuda/scan: cuda/scan.cu $(HEADERS) cuda/hostgate.o $(CUBIOMES) cuda/arch.txt | cuda/lut263.bin
	nvcc -O3 -arch=$(ARCH) -I$(CUB) -Xcompiler -pthread -o $@ cuda/scan.cu cuda/hostgate.o $(CUBIOMES) -lm

# Starts the watchdog in the background, and the watchdog starts the supervisor and the scanner. 
# The scanner checks the range and saves the checkpoint first (--prepare), that way mistakes show up here instead of in the log
run: all
	@if [ -z "$(START)" ] && [ -n "$(COUNT)" ]; then echo "COUNT needs a START too, eg make run START=0 COUNT=1000000000000"; exit 1; fi
	@PROJECT=$$(pwd -P); . ./cuda/procs.sh; \
	if [ -n "$$(ours -f '^bash \./cuda/watchdog\.sh')$$(ours -f '^bash .*supervise\.sh')$$(ours -x scan)" ]; then echo "JUNO is already running here, make stop first if you want to change something"; exit 1; fi
	@if pgrep -x scan > /dev/null; then echo "Something called scan is already running on this computer, stop that first!"; exit 1; fi
	@cd cuda && ./scan $${CUSTOM_SEED:+--custom-seed "$$CUSTOM_SEED"} $(OPTIONS) --prepare $(if $(START),$(START) $(COUNT),run)
	@rm -f results/STOP_GPU results/STOP_WATCHDOG
	@nohup bash ./cuda/watchdog.sh $(OPTIONS) > /dev/null 2>&1 &
	@echo "JUNO is running in the background. make log shows the new records, make status shows how it's going and make stop stops it."

# make says Interrupt when you Ctrl+C out of the log and theres no way to make it stop doing that. whatever
log:
	@MIN_SENTS="$(MIN_SENTS)" MIN_ARBITRATIONS="$(MIN_ARBITRATIONS)" bash ./cuda/log.sh

score: all
	@if [ -z "$(SEED)" ]; then echo "Usage: make score SEED=<seed>"; exit 1; fi
	cd cuda && ./scan score $(SEED)

top: all
	@if [ -z "$(N)" ]; then echo "Usage: make top N=<number> [BY=arbitrations]"; exit 1; fi
	cd cuda && ./scan top $(N) $(BY)

# Scores the example seed from the README on the CPU, we know what its score should be
test: all
	@echo "Scoring seed 3549742502234867244 on the CPU, this takes a few seconds..."
	@RESULT=$$(cd cuda && ./scan score 3549742502234867244 | grep '^Seed'); echo "$$RESULT"; \
	if echo "$$RESULT" | grep -q 'SENTS 0.910265368  ARBITRATIONS 82.850096'; then echo "PASS"; \
	else echo "FAIL, it should have been SENTS 0.910265368 and ARBITRATIONS 82.850096"; exit 1; fi

status:
	@bash ./cuda/status.sh

# Stops the watchdog, the supervisor and the scanner
# The scripts go first (so nothing restarts the scanner), then the scanner gets SIGTERM, which makes it save and stop
stop:
	@bash ./cuda/stop.sh

clean:
	rm -f cuda/scan cuda/genlut cuda/*.o cuda/lut263.bin cuda/lut263.h
