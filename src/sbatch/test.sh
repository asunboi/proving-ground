#!/bin/bash
apptainer exec /tmp/applications/state/0.9.14/bin/state.sif bash -lc '
  echo "Inside container:"
  pwd
  echo "Root:"
  ls /
  echo "Check /gpfs:"
  ls /gpfs || echo "/gpfs does not exist"
  echo "Check your TOML:"
  ls -l /gpfs/home/asun/jin_lab/perturbench/studies/storm/outputs/test/storm/2025-11-17_17-34-20/toml/test_qual_high_amt_high.toml || echo "not visible in container"
'