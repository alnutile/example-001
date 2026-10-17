#!/bin/bash
# usage: run.sh verifier|extractor vN [vM...]
kind=$1; shift; args=""
for v in "$@"; do args="$args --prompts file://prompt_fns.py:${kind}_$v"; done
cd /workspace/promptfoo-crosscheck
PROMPTFOO_PYTHON=$PWD/.venv/bin/python PROMPTFOO_DISABLE_TELEMETRY=1 npx promptfoo eval -c $kind.yaml $args -j 2 --no-cache --no-progress-bar -o results/${kind}_$(echo $@|tr ' ' _).json >/tmp/pf_$kind.log 2>&1
.venv/bin/python summarize.py results/${kind}_$(echo $@|tr ' ' _).json
