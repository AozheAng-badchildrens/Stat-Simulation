# :bar_chart: Stat Simulation Lab

Interactive probability and statistics visualizations built with **Python**, **NumPy**, **SciPy**, and **Matplotlib**.

> Goal: turn abstract statistical theorems into visuals you can see, tweak, and eventually run in real time.

## :sparkles: Highlights

- :game_die: Generate and visualize core distributions (`uniform`, `gamma`, `chi-squared`, `poisson`, etc.)
- :brain: Explore the **Central Limit Theorem (CLT)** with static and animated sampling distributions
- :chart_with_upwards_trend: Overlay theoretical normal approximations on empirical histograms
- :gear: Designed as a foundation for a larger full-stack stats visualizer

## :world_map: Project Roadmap

- [x] Basic distribution sampling + plotting
- [x] CLT static visualization for multiple distributions
- [x] CLT animation with incremental sampling updates
- [ ] Custom discrete / continuous distributions
- [ ] MLE-based distribution fitting for observed data
- [ ] Biased vs unbiased estimator visualizations
- [ ] Consistency and variance convergence demos
- [ ] CRLB intuition and visual lower-bound demos
- [ ] Linear combinations of random variables (e.g., chi-squared sums)
- [ ] Frontend + backend app architecture

## :test_tube: Current CLT Demo

The CLT animation mode repeatedly samples from a base distribution, computes sample means, and updates a histogram over time.
As sample size `n` grows, the sampling distribution of the mean approaches a normal curve.

## :rocket: Quick Start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install numpy scipy matplotlib
python matplotlibfun.py
```

## :hammer_and_wrench: Config You Can Tweak

In `matplotlibfun.py`, adjust:

- `dist_name`: base distribution (`"uniform"`, `"gamma"`, `"chi2"`, `"poisson"`)
- `n`: sample size per replicate
- `total_reps`: total number of replicate samples
- `chunk_size`: how many new sample means are added per animation frame
- `bins`: histogram granularity

## :file_folder: Main File

- `matplotlibfun.py` - distribution samplers, CLT static plots, and CLT animation.

## :compass: Vision

This starts as a notebook-style experiment and grows into a full interactive statistics playground:

- real-time theorem visualizations
- parameter controls and live updates
- upload-your-data workflows
- fit diagnostics and model comparison

## :handshake: Contributing

Ideas and PRs are welcome, especially around:

- additional distributions
- better animation UX
- numerical stability / performance
- educational annotations on plots

## :page_facing_up: License

No license has been added yet.
