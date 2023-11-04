#!/bin/bash

for i in {1..5}; do
   echo "Seed $i"
   python run_survival.py encoder.base.method=lstm experiment.seed="$i";
   # python run_survival.py experiment.seed="$i" encoder.base.method=lstm;
   # python run_survival.py experiment.seed="$i" encoder.base.method=cnn;
   # python run_survival.py experiment.seed="$i" encoder.base.method=average;
done
