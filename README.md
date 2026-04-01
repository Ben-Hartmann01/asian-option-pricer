# Asian Option Pricer

This project implements Quasi-Monte Carlo simulation for pricing Asian options under the Black–Scholes model (geometric Brownian motion) using PCA for dimension reduction.

The pricer supports:
- Asian call options
- Asian put options
- Asian barrier call options
- Asian barrier put options

## Model
The underlying asset is modeled under Black–Scholes dynamics (GBM).  
Pricing is performed with Quasi-Monte Carlo simulation, and PCA is used to improve efficiency in the path generation.

## Run
To run the pricer, execute:

```bash
python asian_option_pricer.py


