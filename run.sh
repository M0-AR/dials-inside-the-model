#!/bin/bash
# Reproduce all results. Fast path (~2-3 min). Use --full to include part8 scaling sweep (~+2 min).
set -e
cd "$(dirname "$0")"
echo "[1/8] part1 model"; python3 parts/part1_model.py
echo "[2/8] part2 neurons"; python3 parts/part2_neurons.py
echo "[3/8] part3 superposition"; python3 parts/part3_superposition.py
echo "[4/8] part4 sae toy"; python3 parts/part4_sae_toy.py
echo "[5/8] part5 sae real"; python3 parts/part5_sae_real.py
echo "[6/8] part6 read"; python3 parts/part6_read.py
echo "[7/8] part7 steer"; python3 parts/part7_steer.py
if [ "$1" = "--full" ]; then
  echo "[8/8] part8 scaling"; python3 parts/part8_scaling.py
else
  echo "[8/8] part8 SKIPPED (run with --full)"
fi
echo "[bench]"; python3 benchmarks/metrics.py
echo "[live]"; python3 verification/live_public_verify.py
echo "ALL DONE. See results/ + README.md"
