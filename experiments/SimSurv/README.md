To run the Simulated Survival experiment

```console

python run_survival.py encoder.base.method=waveLSTM experiment.seed=1 experiment.output_dir=<path_to_out_dir>
```

Results in the manuscript are obtained using seeds 1 to 5. 

To perform the benchmark examples, set the base method argument to one of `LSTM`, `cnn`, or `avg`, using the same seeds. Other configuration choices also be made, as shown in the configurations.
