#!/bin/sh
# Full disc-model pipeline after the solution table exists (scripts/build_table.py).
set -e
cd "$(dirname "$0")/.."
python3 scripts/run_experiments.py --geometry disc --methods pen mlp --seeds 0 1 2 3 4
python3 scripts/summarize.py --geometry disc
python3 scripts/run_cv.py --geometry disc --schemes lomo lofo --methods fdreg mlp pen
python3 scripts/summarize_cv.py --geometry disc
python3 scripts/run_experiments.py --geometry disc --methods pen mlp --seeds 0 1 2 3 4 --noise 0.01
python3 scripts/run_experiments.py --geometry disc --methods pen mlp --seeds 0 1 2 3 4 --noise 0.02
python3 scripts/identifiability.py --geometry disc
echo PIPELINE_DONE
