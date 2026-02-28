import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------
from pathlib import Path

# Project root = directory where main.py lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Input file
CSV_PATH = PROJECT_ROOT / "data" / "open_meteo_51.78N10.35E563m.csv"

MIN_SAMPLES = 20             # minimum data points required

# Candidate distributions
DISTRIBUTIONS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "exponential": stats.expon,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "student_t": stats.t,
    "betta": stats.beta
}

# -----------------------------
# FUNCTIONS
# -----------------------------
def fit_distribution(data, dist):
    params = dist.fit(data)
    loglik = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * loglik
    ks_stat, ks_p = stats.kstest(data, dist.name, args=params)
    return aic, ks_stat, ks_p, params


def analyze_variable(data):
    results = []

    for name, dist in DISTRIBUTIONS.items():
        try:
            aic, ks_stat, ks_p, params = fit_distribution(data, dist)
            results.append({
                "distribution": name,
                "AIC": aic,
                "KS_stat": ks_stat,
                "KS_pvalue": ks_p,
                "params": params
            })
        except Exception:
            continue

    if not results:
        return None

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("AIC")
    return results_df


# -----------------------------
# MAIN
# -----------------------------
df = pd.read_csv(CSV_PATH)

summary = []

for col in df.columns:
    if not np.issubdtype(df[col].dtype, np.number):
        continue

    data = df[col].dropna()

    if len(data) < MIN_SAMPLES:
        continue

    results_df = analyze_variable(data)

    if results_df is None:
        continue

    best = results_df.iloc[0]

    summary.append({
        "variable": col,
        "best_distribution": best["distribution"],
        "AIC": best["AIC"],
        "KS_pvalue": best["KS_pvalue"],
        "parameters": best["params"]
    })

summary_df = pd.DataFrame(summary)

print("\n===== BEST FIT DISTRIBUTIONS =====\n")
print(summary_df)

# Optional: save results
summary_df.to_csv("distribution_fitting_results.csv", index=False)
