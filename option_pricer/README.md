# Asian & Barrier Option Pricing with Monte Carlo and Quasi-Monte Carlo

This repository implements pricing engines for path-dependent options under the Black–Scholes model using advanced simulation techniques.

The focus is on comparing **Monte Carlo (MC)** and **Quasi-Monte Carlo (QMC)** methods, including **PCA-based variance reduction** and **randomized QMC (RQMC)**.

---

## Features

* Arithmetic and geometric **Asian options**
* **Asian barrier options** (knock-in / knock-out)
* Standard **Monte Carlo (MC)**
* **Sobol Quasi-Monte Carlo (QMC)**
* **PCA construction** for Brownian motion
* **Randomized QMC (RQMC)** for error estimation
* Convergence analysis and visualization

---

## Project Structure

```
project/
│
├── engines/
│   ├── asian_option_qmc.py
│   ├── asian_barrier_option_engine.py
│
├── experiments/
│   ├── convergence_compare.py
│   ├── convergence_compare_barrier.py
│   ├── plot_convergence.py
│   ├── plot_convergence_barrier.py
│
├── requirements.txt
├── README.md
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run convergence comparison (Asian option)

```bash
python experiments/convergence_compare.py
```

### Run convergence comparison (Barrier option)

```bash
python experiments/convergence_compare_barrier.py
```

### Plot convergence (Asian option)

```bash
python experiments/plot_convergence.py
```

### Plot convergence (Barrier option)

```bash
python experiments/plot_convergence_barrier.py
```

---

## Key Results & Insights

### Asian Options

* QMC significantly improves convergence over MC
* PCA reduces effective dimension and enhances QMC performance
* Convergence can approach **O(N⁻¹)** in favorable cases

### Barrier Options

* Payoff discontinuity reduces effectiveness of QMC
* PCA provides limited improvement
* Convergence behaves closer to standard MC (**O(N⁻¹ᐟ²)**)

---

## Methodology

* Underlying model: **Black–Scholes (GBM)**
* Path simulation via:

  * Standard Brownian increments
  * PCA-based Brownian construction
* Low-discrepancy sampling via:

  * Sobol sequences
  * Scrambling for RQMC

---

## Convergence Analysis

The project includes log-log plots of standard error vs number of paths.

Typical observations:

* MC slope ≈ -1/2
* QMC + PCA shows improved convergence for smooth payoffs
* Barrier options highlight limitations of QMC for discontinuous payoffs

---

## Requirements

* Python 3.10+
* numpy
* scipy
* matplotlib

---

## Notes

* Geometric Asian option provides a useful benchmark
* Barrier options require full path monitoring
* RQMC enables statistically meaningful error estimates

---

## Purpose

This project is intended as a **quantitative finance / numerical methods study** demonstrating:

* variance reduction techniques
* high-dimensional integration
* practical differences between MC and QMC methods

---

## Author

Feel free to use, modify, or extend this code for research or learning purposes.
