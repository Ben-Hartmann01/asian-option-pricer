from __future__ import annotations

"""
Convergence comparison script for the Asian barrier option engine.

How to use:
1. Save your original pricing code in a file, for example: asian_barrier_option_engine.py
2. Save this file next to it as: convergence_barrier_compare.py
3. Run:
       python convergence_compare_barrier.py

This script compares:
- standard Monte Carlo
- Sobol QMC without PCA
- Sobol QMC with full PCA
- Sobol QMC with truncated PCA
- randomized QMC (RQMC) with independent scrambles
"""

from engines.asian_barrier_option_engine import (
    AsianBarrierOptionSpec,
    MarketParams,
    mc_asian_barrier_price,
    qmc_asian_barrier_price_sobol,
    rqmc_asian_barrier_price_sobol,
)


def _format_row(label: str, n: int, price: float, stderr: float) -> str:
    return f"{label:<28} {n:>10,d}   {price:>12.6f}   {stderr:>12.6f}"


def run_convergence_study() -> str:
    market = MarketParams(s0=100.0, r=0.03, sigma=0.20, t=1.0)
    option = AsianBarrierOptionSpec(
        strike=100.0,
        barrier=120.0,
        n_fixings=52,
        option_type="call",
        average_type="arithmetic",
        barrier_type="up-and-out",
    )

    # Powers of two are especially natural for Sobol.
    path_grid = [2**k for k in range(8, 18)]  # 256 to 131072

    lines: list[str] = []
    lines.append("=" * 88)
    lines.append("CONVERGENCE COMPARISON: MC vs QMC")
    lines.append("=" * 88)
    lines.append("")
    lines.append(f"Spot S0      : {market.s0:.4f}")
    lines.append(f"Strike K     : {option.strike:.4f}")
    lines.append(f"Barrier B    : {option.barrier:.4f}")
    lines.append(f"Rate r       : {market.r:.4%}")
    lines.append(f"Vol sigma    : {market.sigma:.4%}")
    lines.append(f"Maturity T   : {market.t:.4f}")
    lines.append(f"Fixings      : {option.n_fixings}")
    lines.append(f"Option Type  : {option.option_type}")
    lines.append(f"Average Type : {option.average_type}")
    lines.append(f"Barrier Type : {option.barrier_type}")
    lines.append("")

    methods = [
        "MC",
        "QMC Sobol",
        "QMC Sobol + full PCA",
        "QMC Sobol + PCA(8)",
        "RQMC Sobol + PCA(8)",
    ]

    summary: dict[str, list[tuple[int, float, float]]] = {m: [] for m in methods}

    for n_paths in path_grid:
        mc_price, mc_se = mc_asian_barrier_price(
            market=market,
            option=option,
            n_paths=n_paths,
            seed=42,
            use_pca=False,
        )
        summary["MC"].append((n_paths, mc_price, mc_se))

        qmc_plain_price, qmc_plain_se = qmc_asian_barrier_price_sobol(
            market=market,
            option=option,
            n_paths=n_paths,
            use_pca=False,
            scramble=True,
            seed=42,
        )
        summary["QMC Sobol"].append((n_paths, qmc_plain_price, qmc_plain_se))

        qmc_full_price, qmc_full_se = qmc_asian_barrier_price_sobol(
            market=market,
            option=option,
            n_paths=n_paths,
            use_pca=True,
            n_components=option.n_fixings,
            scramble=True,
            seed=42,
        )
        summary["QMC Sobol + full PCA"].append((n_paths, qmc_full_price, qmc_full_se))

        qmc_pca8_price, qmc_pca8_se = qmc_asian_barrier_price_sobol(
            market=market,
            option=option,
            n_paths=n_paths,
            use_pca=True,
            n_components=8,
            scramble=True,
            seed=42,
        )
        summary["QMC Sobol + PCA(8)"].append((n_paths, qmc_pca8_price, qmc_pca8_se))

        # For RQMC, treat n_paths as the number of points per replication.
        rqmc_price, rqmc_se, _ = rqmc_asian_barrier_price_sobol(
            market=market,
            option=option,
            n_paths=n_paths,
            n_replications=16,
            use_pca=True,
            n_components=8,
            base_seed=100,
        )
        summary["RQMC Sobol + PCA(8)"].append((n_paths, rqmc_price, rqmc_se))

    for method in methods:
        lines.append("-" * 88)
        lines.append(method)
        lines.append("-" * 88)
        lines.append(f"{'Method':<28} {'Paths':>10}   {'Price':>12}   {'StdErr':>12}")
        for n_paths, price, se in summary[method]:
            lines.append(_format_row(method, n_paths, price, se))
        lines.append("")

    # Simple final comparison at the largest path count.
    lines.append("=" * 88)
    lines.append("FINAL STDERR COMPARISON AT LARGEST PATH COUNT")
    lines.append("=" * 88)
    max_n = path_grid[-1]
    for method in methods:
        n_paths, price, se = summary[method][-1]
        assert n_paths == max_n
        lines.append(f"{method:<28} N={n_paths:>10,d}   Price={price:>10.6f}   StdErr={se:>10.6f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- MC should typically converge like O(N^(-1/2)).")
    lines.append("- QMC often improves convergence, especially with low effective dimension.")
    lines.append("- PCA can help QMC by concentrating variance in earlier coordinates.")
    lines.append("- RQMC gives a more meaningful error estimate through independent replications.")
    lines.append("- Barrier features make convergence less smooth than for plain Asian options.")

    return "\n".join(lines)


if __name__ == "__main__":
    print(run_convergence_study())