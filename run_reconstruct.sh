#!/usr/bin/env bash

for step in {0..50000..1000}; do
  #python3 reconstruction.py $step
  python3 mask_predict.py $step
done
