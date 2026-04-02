"""
Asian barrier option pricing engine under Black-Scholes / Brownian motion.
Supports:
- standard Monte Carlo
- Sobol quasi-Monte Carlo
- Sobol random quasi-Monte Carlo
- PCA construction for Brownian paths
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Literal

import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm, qmc

OptionType = Literal["call", "put"]
AverageType = Literal["arithmetic", "geometric"]
BarrierType = Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"]


@dataclass
class MarketParams:
    s0: float = 100.0
    r: float = 0.03
    sigma: float = 0.20
    t: float = 1.0


@dataclass
class AsianBarrierOptionSpec:
    strike: float = 100.0
    barrier: float = 120.0
    n_fixings: int = 52
    option_type: OptionType = "call"
    average_type: AverageType = "arithmetic"
    barrier_type: BarrierType = "up-and-out"


def black_scholes_european_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    option_type: OptionType = "call",
) -> float:
    if t <= 0:
        return max(s0 - k, 0.0) if option_type == "call" else max(k - s0, 0.0)

    vol_sqrt_t = sigma * sqrt(t)
    d1 = (log(s0 / k) + (r + 0.5 * sigma**2) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "call":
        return s0 * norm.cdf(d1) - k * exp(-r * t) * norm.cdf(d2)
    return k * exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1)


def geometric_asian_price_bs(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_fixings: int,
    option_type: OptionType = "call",
) -> float:
    if n_fixings < 1:
        raise ValueError("n_fixings must be >= 1")

    n = n_fixings
    a = (n + 1.0) / (2.0 * n)
    b = (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n**2)

    mu_g = log(s0) + (r - 0.5 * sigma**2) * t * a
    var_g = sigma**2 * t * b
    sigma_g = sqrt(var_g)

    d1 = (mu_g - log(k) + var_g) / sigma_g
    d2 = d1 - sigma_g
    discounted_forward_component = exp(-r * t + mu_g + 0.5 * var_g)

    if option_type == "call":
        return discounted_forward_component * norm.cdf(d1) - exp(-r * t) * k * norm.cdf(d2)
    return exp(-r * t) * k * norm.cdf(-d2) - discounted_forward_component * norm.cdf(-d1)


def brownian_covariance(times: np.ndarray) -> np.ndarray:
    return np.minimum.outer(times, times)


def pca_transform_for_brownian(times: np.ndarray, n_components: int | None = None) -> np.ndarray:
    cov = brownian_covariance(times)
    eigvals, eigvecs = eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    return eigvecs * np.sqrt(eigvals)


def simulate_gbm_paths_from_normals(
    z: np.ndarray,
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_fixings: int,
    use_pca: bool = True,
    n_components: int | None = None,
) -> np.ndarray:
    times = np.linspace(t / n_fixings, t, n_fixings)

    if use_pca:
        loadings = pca_transform_for_brownian(times, n_components=n_components)
        if z.shape[1] != loadings.shape[1]:
            raise ValueError(
                f"z has dimension {z.shape[1]} but PCA loadings require {loadings.shape[1]} dimensions."
            )
        w = z @ loadings.T
    else:
        if z.shape[1] != n_fixings:
            raise ValueError("Without PCA, z must have dimension n_fixings.")
        dt = t / n_fixings
        w = np.cumsum(np.sqrt(dt) * z, axis=1)

    drift = (r - 0.5 * sigma**2) * times
    log_s = np.log(s0) + drift[None, :] + sigma * w
    return np.exp(log_s)


def asian_barrier_payoff(
    paths: np.ndarray,
    strike: float,
    barrier: float,
    option_type: OptionType = "call",
    average_type: AverageType = "arithmetic",
    barrier_type: BarrierType = "up-and-out",
    s0: float | None = None,
) -> np.ndarray:
    if average_type == "arithmetic":
        avg = paths.mean(axis=1)
    elif average_type == "geometric":
        avg = np.exp(np.log(paths).mean(axis=1))
    else:
        raise ValueError("average_type must be 'arithmetic' or 'geometric'.")

    if option_type == "call":
        vanilla = np.maximum(avg - strike, 0.0)
    else:
        vanilla = np.maximum(strike - avg, 0.0)

    if s0 is not None:
        s0_col = np.full((paths.shape[0], 1), s0)
        monitored = np.concatenate([s0_col, paths], axis=1)
    else:
        monitored = paths

    path_max = monitored.max(axis=1)
    path_min = monitored.min(axis=1)

    if barrier_type == "up-and-out":
        active = path_max < barrier
    elif barrier_type == "down-and-out":
        active = path_min > barrier
    elif barrier_type == "up-and-in":
        active = path_max >= barrier
    elif barrier_type == "down-and-in":
        active = path_min <= barrier
    else:
        raise ValueError(
            "barrier_type must be 'up-and-out', 'down-and-out', 'up-and-in', or 'down-and-in'."
        )

    return vanilla * active.astype(float)


def discounted_payoffs_from_normals(
    z: np.ndarray,
    market: MarketParams,
    option: AsianBarrierOptionSpec,
    use_pca: bool = False,
    n_components: int | None = None,
) -> np.ndarray:
    paths = simulate_gbm_paths_from_normals(
        z=z,
        s0=market.s0,
        r=market.r,
        sigma=market.sigma,
        t=market.t,
        n_fixings=option.n_fixings,
        use_pca=use_pca,
        n_components=n_components,
    )
    payoffs = asian_barrier_payoff(
        paths=paths,
        strike=option.strike,
        barrier=option.barrier,
        option_type=option.option_type,
        average_type=option.average_type,
        barrier_type=option.barrier_type,
        s0=market.s0,
    )
    return np.exp(-market.r * market.t) * payoffs


def mc_asian_barrier_price(
    market: MarketParams,
    option: AsianBarrierOptionSpec,
    n_paths: int = 100_000,
    seed: int = 42,
    use_pca: bool = False,
    n_components: int | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = option.n_fixings if not use_pca else (n_components or option.n_fixings)
    z = rng.standard_normal(size=(n_paths, d))

    discounted = discounted_payoffs_from_normals(
        z=z,
        market=market,
        option=option,
        use_pca=use_pca,
        n_components=n_components,
    )
    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price, stderr


def sobol_normal_draws(
    n_paths: int,
    dim: int,
    scramble: bool = True,
    seed: int | None = 42,
) -> np.ndarray:
    sampler = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    m = int(np.ceil(np.log2(max(1, n_paths))))
    u = sampler.random_base2(m=m)[:n_paths]

    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)
    return norm.ppf(u)


def qmc_asian_barrier_price_sobol(
    market: MarketParams,
    option: AsianBarrierOptionSpec,
    n_paths: int = 131_072,
    use_pca: bool = True,
    n_components: int | None = None,
    scramble: bool = True,
    seed: int | None = 42,
) -> tuple[float, float]:
    d = option.n_fixings if not use_pca else (n_components or option.n_fixings)
    z = sobol_normal_draws(n_paths=n_paths, dim=d, scramble=scramble, seed=seed)

    discounted = discounted_payoffs_from_normals(
        z=z,
        market=market,
        option=option,
        use_pca=use_pca,
        n_components=n_components,
    )
    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price, stderr


def rqmc_asian_barrier_price_sobol(
    market: MarketParams,
    option: AsianBarrierOptionSpec,
    n_paths: int = 8192,
    n_replications: int = 16,
    use_pca: bool = True,
    n_components: int | None = None,
    base_seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Randomized QMC estimator using independent scrambled Sobol replications.

    Returns:
        mean_estimate, stderr_across_replications, replication_estimates
    """
    d = option.n_fixings if not use_pca else (n_components or option.n_fixings)
    estimates = np.empty(n_replications, dtype=float)

    for rep in range(n_replications):
        z = sobol_normal_draws(
            n_paths=n_paths,
            dim=d,
            scramble=True,
            seed=base_seed + rep,
        )
        discounted = discounted_payoffs_from_normals(
            z=z,
            market=market,
            option=option,
            use_pca=use_pca,
            n_components=n_components,
        )
        estimates[rep] = discounted.mean()

    mean_estimate = float(estimates.mean())
    stderr = float(estimates.std(ddof=1) / np.sqrt(n_replications))
    return mean_estimate, stderr, estimates


def demo() -> str:
    market = MarketParams(s0=100.0, r=0.03, sigma=0.20, t=1.0)
    option = AsianBarrierOptionSpec(
        strike=100.0,
        barrier=120.0,
        n_fixings=52,
        option_type="call",
        average_type="arithmetic",
        barrier_type="up-and-out",
    )

    european_call = black_scholes_european_price(
        market.s0, option.strike, market.r, market.sigma, market.t, "call"
    )
    geometric_asian = geometric_asian_price_bs(
        market.s0, option.strike, market.r, market.sigma, market.t, option.n_fixings, "call"
    )

    mc_price, mc_stderr = mc_asian_barrier_price(
        market=market,
        option=option,
        n_paths=100_000,
        seed=42,
        use_pca=False,
    )

    qmc_price_full, qmc_stderr_full = qmc_asian_barrier_price_sobol(
        market=market,
        option=option,
        n_paths=131_072,
        use_pca=True,
        n_components=option.n_fixings,
        scramble=True,
        seed=42,
    )

    qmc_price_trunc, qmc_stderr_trunc = qmc_asian_barrier_price_sobol(
        market=market,
        option=option,
        n_paths=131_072,
        use_pca=True,
        n_components=8,
        scramble=True,
        seed=42,
    )

    rqmc_price, rqmc_stderr, rqmc_estimates = rqmc_asian_barrier_price_sobol(
        market=market,
        option=option,
        n_paths=8192,
        n_replications=16,
        use_pca=True,
        n_components=8,
        base_seed=42,
    )

    lines = [
        "=" * 50,
        "ASIAN BARRIER OPTION SIMULATOR",
        "=" * 50,
        f"Spot S0        : {market.s0:.4f}",
        f"Strike K       : {option.strike:.4f}",
        f"Barrier B      : {option.barrier:.4f}",
        f"Rate r         : {market.r:.4%}",
        f"Vol sigma      : {market.sigma:.4%}",
        f"Maturity T     : {market.t:.4f}",
        f"Fixings        : {option.n_fixings}",
        f"Option Type    : {option.option_type}",
        f"Average Type   : {option.average_type}",
        f"Barrier Type   : {option.barrier_type}",
        "",
        f"European Black-Scholes call benchmark          : {european_call:10.6f}",
        f"Geometric Asian closed-form benchmark          : {geometric_asian:10.6f}",
        f"Asian barrier MC price                         : {mc_price:10.6f}  (SE {mc_stderr:.6f})",
        f"Asian barrier Sobol-QMC + full PCA             : {qmc_price_full:10.6f}  (pseudo-SE {qmc_stderr_full:.6f})",
        f"Asian barrier Sobol-QMC + truncated PCA(8)     : {qmc_price_trunc:10.6f}  (pseudo-SE {qmc_stderr_trunc:.6f})",
        f"Asian barrier RQMC + truncated PCA(8)          : {rqmc_price:10.6f}  (SE {rqmc_stderr:.6f})",
        "",
        "Notes:",
        "- European BS price is not a benchmark for the Asian barrier option value.",
        "- Geometric Asian closed-form is only a rough reference here, not a true barrier benchmark (it has no barrier).",
        "- Barrier features make convergence more sensible than for plain Asian options.",
        "- Single-run Sobol error figures are pseudo-SEs, not rigorous MC-style standard errors due to correlation."
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo())