from __future__ import annotations

"""
Plot convergence of applied methods (Asian barrier option).

Produces log-log plots of standard error vs number of paths.
"""

import numpy as np
import matplotlib.pyplot as plt

from engines.asian_barrier_option_engine import (
    AsianBarrierOptionSpec,
    MarketParams,
    mc_asian_barrier_price,
    qmc_asian_barrier_price_sobol,
    rqmc_asian_barrier_price_sobol,
)


def run_convergence_data():
    market = MarketParams(s0=100.0, r=0.03, sigma=0.20, t=1.0)
    option = AsianBarrierOptionSpec(
        strike=100.0,
        barrier=120.0,
        n_fixings=52,
        option_type="call",
        average_type="arithmetic",
        barrier_type="up-and-out",
    )

    path_grid = np.array([2**k for k in range(8, 18)])

    mc_errors = []
    qmc_errors = []
    qmc_pca_errors = []
    rqmc_errors = []

    for n_paths in path_grid:
        _, mc_se = mc_asian_barrier_price(
            market=market,
            option=option,
            n_paths=n_paths,
            seed=42,
            use_pca=False,
        )
        mc_errors.append(mc_se)

        _, qmc_se = qmc_asian_barrier_price_sobol(
            market=market,
            option=option,
            n_paths=n_paths,
            use_pca=False,
            scramble=True,
            seed=42,
        )
        qmc_errors.append(qmc_se)

        _, qmc_pca_se = qmc_asian_barrier_price_sobol(
            market=market,
            option=option,
            n_paths=n_paths,
            use_pca=True,
            n_components=8,
            scramble=True,
            seed=42,
        )
        qmc_pca_errors.append(qmc_pca_se)

        _, rqmc_se, _ = rqmc_asian_barrier_price_sobol(
            market=market,
            option=option,
            n_paths=n_paths,
            n_replications=16,
            use_pca=True,
            n_components=8,
            base_seed=100,
        )
        rqmc_errors.append(rqmc_se)

    return path_grid, np.array(mc_errors), np.array(qmc_errors), np.array(qmc_pca_errors), np.array(rqmc_errors)


def plot_convergence():
    N, mc, qmc, qmc_pca, rqmc = run_convergence_data()

    plt.figure()

    plt.loglog(N, mc, marker="o", label="MC")
    plt.loglog(N, qmc, marker="o", label="QMC Sobol")
    plt.loglog(N, qmc_pca, marker="o", label="QMC + PCA(8)")
    plt.loglog(N, rqmc, marker="o", label="RQMC + PCA(8)")

    ref = mc[0] * (N / N[0]) ** (-0.5)
    plt.loglog(N, ref, linestyle="--", label="O(N^{-1/2}) reference")

    plt.xlabel("Number of paths (log scale)")
    plt.ylabel("Standard error (log scale)")
    plt.title("Convergence: MC vs QMC (Asian Barrier Option)")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    plot_convergence()