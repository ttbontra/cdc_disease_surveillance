from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az


@dataclass(frozen=True)
class BayesHierarchicalResult:
    regions: List[str]
    time_index: List[str]           # dates in training
    posterior_summary: Dict[str, Any]
    idata: Any                      # InferenceData (arviz)


def fit_hierarchical_growth_model(master: pd.DataFrame,
                                  value_col: str = "cases",
                                  min_days: int = 60,
                                  draws: int = 1000,
                                  tune: int = 1000,
                                  target_accept: float = 0.9,
                                  random_seed: int = 42) -> BayesHierarchicalResult:
    """
    Hierarchical log-growth model (comparator to SEIR):
      y_{r,t} ~ NegBinomial(mu_{r,t}, alpha)
      log(mu_{r,t}) = a_r + b_r * t
      a_r ~ Normal(a0, sigma_a)
      b_r ~ Normal(b0, sigma_b)

    Pros:
      - fast-ish
      - clean uncertainty
      - borrows strength across regions

    This is not mechanistic SEIR. It is a probabilistic comparator for forecasting.
    """
    df = master.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"])

    # align to common window across regions
    # use last min_days days available per region
    frames = []
    for region, g in df.groupby("region"):
        g = g.dropna(subset=[value_col]).copy()
        if len(g) < min_days:
            continue
        g = g.tail(min_days)
        frames.append(g)

    if not frames:
        raise ValueError(f"Not enough data to fit Bayesian model (need >= {min_days} days per region).")

    data = pd.concat(frames, ignore_index=True)
    regions = sorted(data["region"].unique().tolist())

    # Create integer time index per region aligned by date order
    # We'll use a global t (0..min_days-1) shared across regions (since we truncated to tail(min_days)).
    # Also create region codes.
    data = data.sort_values(["region", "date"])
    data["t"] = data.groupby("region").cumcount().astype(int)
    region_codes = {r: i for i, r in enumerate(regions)}
    data["r"] = data["region"].map(region_codes).astype(int)

    y = data[value_col].astype(int).clip(lower=0).values
    t = data["t"].values
    r = data["r"].values
    R = len(regions)

    # For labeling
    time_index = (
        data[data["region"] == regions[0]]["date"]
        .dt.date.astype(str)
        .tolist()
    )

    with pm.Model() as model:
        # global priors
        a0 = pm.Normal("a0", mu=np.log(np.mean(y + 1)), sigma=2.0)
        b0 = pm.Normal("b0", mu=0.0, sigma=0.1)

        sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.1)

        # region random effects
        a_r = pm.Normal("a_r", mu=a0, sigma=sigma_a, shape=R)
        b_r = pm.Normal("b_r", mu=b0, sigma=sigma_b, shape=R)

        # overdispersion
        alpha = pm.HalfNormal("alpha", sigma=10.0)

        # linear predictor
        eta = a_r[r] + b_r[r] * t
        mu = pm.Deterministic("mu", pm.math.exp(eta))

        # NegBinomial parameterization in PyMC: mu + alpha
        obs = pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            chains=2,
            cores=1,
            progressbar=False,
        )

    summary = az.summary(idata, var_names=["a0", "b0", "sigma_a", "sigma_b", "alpha"], kind="stats").to_dict()
    return BayesHierarchicalResult(
        regions=regions,
        time_index=time_index,
        posterior_summary=summary,
        idata=idata,
    )


def forecast_posterior_predictive(result: BayesHierarchicalResult,
                                  horizon_days: int = 28,
                                  quantiles=(0.1, 0.5, 0.9)) -> pd.DataFrame:
    """
    Produce forecast bands per region from posterior samples:
      mu_{r,t} = exp(a_r + b_r * t)
    Returns a tidy table with quantiles for each region and date index (history+future).
    """
    idata = result.idata
    posterior = idata.posterior

    a_r = posterior["a_r"].stack(sample=("chain", "draw")).values  # shape (R, S)
    b_r = posterior["b_r"].stack(sample=("chain", "draw")).values  # shape (R, S)

    R, S = a_r.shape
    T_hist = len(result.time_index)
    T_all = T_hist + horizon_days

    # t grid
    t = np.arange(T_all, dtype=float)  # 0..T_all-1
    rows = []
    for ri, region in enumerate(result.regions):
        # samples: mu(t) for each sample
        mu_samples = np.exp(a_r[ri, :].reshape(1, S) + b_r[ri, :].reshape(1, S) * t.reshape(T_all, 1))
        # quantiles across samples
        qs = np.quantile(mu_samples, q=quantiles, axis=1)
        for ti in range(T_all):
            date_label = result.time_index[ti] if ti < T_hist else f"FUTURE+{ti - T_hist + 1}"
            rows.append({
                "region": region,
                "t": int(ti),
                "date_label": date_label,
                **{f"q{int(q*100)}": float(qs[j, ti]) for j, q in enumerate(quantiles)}
            })

    return pd.DataFrame(rows)
