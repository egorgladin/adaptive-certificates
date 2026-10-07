# Adaptive Certificates

Code accompanying *Dimension-Adaptive Accuracy Certificates for Convex Optimization*. Licensed under [MIT](LICENSE).

Use Python 3.12 and run the commands from this directory. Use separate environments for the two dependency lists.

**Certificate bandwidth.** Requires an NVIDIA GPU and CUDA-enabled PyTorch (tested with a PyTorch 2.6 development build/CUDA 12.1).

```sh
pip install -r requirements.txt
python -m experiment.study
```

The run downloads the four UCI datasets, selects parameters, runs all comparisons and produces CSV tables and PDF figures in `output/`. Stages can also be run separately: `prepare`, `tuning`, `pilot`, `confirmation`, `sensitivity`, `results`. Existing runs are not overwritten; timings depend on hardware.

**Adaptive cutting plane.** CPU only.

```sh
pip install -r adaptive_cp/requirements.txt
python -m adaptive_cp.runner grid --output output/adaptive_cp --seconds 180
python -m adaptive_cp.reporting --results output/adaptive_cp
```

This runs all 31 comparisons and generates the figure and tables. Optional numerical checks: `python -m adaptive_cp.check` and `python -m adaptive_cp.check --grid output/adaptive_cp`. Each grid requires a new output directory.
