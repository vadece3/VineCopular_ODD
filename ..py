import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from copulas.univariate import (
    GammaUnivariate,
    GaussianUnivariate,
    BetaUnivariate,
    UniformUnivariate
)


###############################################################################
# Distribution detection logic (from Code 2)
###############################################################################
DISTRIBUTIONS = {
    "normal": stats.norm,
    "lognormal": stats.lognorm,
    "exponential": stats.expon,
    "gamma": stats.gamma,
    "weibull": stats.weibull_min,
    "student_t": stats.t,
    "beta": stats.beta
}

def fit_distribution(data, dist):
    params = dist.fit(data)
    loglik = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * loglik
    return aic, params

def detect_best_distribution(data):
    results = []

    for name, dist in DISTRIBUTIONS.items():
        try:
            aic, params = fit_distribution(data, dist)
            results.append((name, aic, params))
        except Exception:
            continue

    if not results:
        return None

    results.sort(key=lambda x: x[1])
    return results[0][0]  # best distribution name

###############################################################################
# Map SciPy distributions → Copulas marginals
###############################################################################
COPULA_MARGINAL_MAP = {
    "normal": GaussianUnivariate,
    "student_t": GaussianUnivariate,
    "lognormal": GaussianUnivariate,
    "gamma": GammaUnivariate,
    "exponential": GammaUnivariate,
    "weibull": GammaUnivariate,
    "beta": BetaUnivariate
}

###############################################################################
# Safe vectorized wrappers for copulas univariate models
###############################################################################
def safe_cdf(marginal, x):
    x = np.asarray(x).reshape(-1, 1)
    return marginal.cumulative_distribution(x).flatten()

def safe_ppf(marginal, u):
    u = np.asarray(u).reshape(-1, 1)
    return marginal.percent_point(u).flatten()

###############################################################################
# Step 1: Load data
###############################################################################
from pathlib import Path

# Project root = directory where main.py lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Input file
input_path = PROJECT_ROOT / "data" / "open_meteo_51.78N10.35E563m.csv"

# Output file
output_path = PROJECT_ROOT / "output" / "predicted_odd_samples.csv"

df = pd.read_csv(input_path)

# df = df[
#     ['Rainmm', 'WindSpeedkmPerH', 'TemperatureDegreeCelcius',
#      'CloudCoverPercentage', 'Snowfallcm']
# ].dropna()

###############################################################################
# Step 1b: Column-wise extreme selection (TOP n rows per column)
###############################################################################
total_rows = len(df)
n = total_rows // 4

print(f"\n Extreme selection: n = {n} rows per column\n")

subset_dfs = {}

for col in df.columns:
    subset_dfs[col] = (
        df.sort_values(by=col, ascending=False)
          .head(n)
          .reset_index(drop=True)
    )

    print(f"  Subset created for column: {col}")


###############################################################################
# Steps 2–5 (REPEATED per column subset)
###############################################################################
all_uniform_data = []

final_marginal_models = {}
final_distribution_summary = {}

for driving_col, sub_df in subset_dfs.items():

    print(f"\n Processing subset driven by: {driving_col}")

    df_sub = sub_df.copy()

    # ------------------ Step 2:  Physically-aware preprocessing (UNIT CONSISTENT) ------------------

    # --- Rain (mm): non-negative, strictly positive for continuous distributions
    df_sub['Rainmm'] = (
        df_sub['Rainmm']
        .clip(lower=0.0)  # physical lower bound
        .replace(0.0, 0.001)  # avoid degenerate PDFs
    )

    # --- Wind speed (km/h): non-negative, allow calm conditions
    df_sub['WindSpeedkmPerH'] = (
        df_sub['WindSpeedkmPerH']
        .clip(lower=0.0)
    )

    # --- Temperature (°C): physically unconstrained, no clipping
    # (negative values allowed, extremes preserved)
    df_sub['TemperatureDegreeCelcius'] = df_sub['TemperatureDegreeCelcius']

    # --- Cloud cover (%): enforce [0, 100], convert to [0, 1] for modeling
    df_sub['CloudCoverPercentage'] = (
            df_sub['CloudCoverPercentage']
            .clip(lower=0.0, upper=100.0) / 100.0
    )

    # --- Snowfall (cm): non-negative, strictly positive for continuous distributions
    df_sub['Snowfallcm'] = (
        df_sub['Snowfallcm']
        .clip(lower=0.0)
        .replace(0.0, 0.001)
    )

    # ------------------ Step 3: Automatic marginal fitting ------------------
    marginal_models = {}
    print("  Marginal Distribution Selection:")

    for col in df_sub.columns:
        data = df_sub[col].values
        best_dist = detect_best_distribution(data)

        marginal_class = COPULA_MARGINAL_MAP.get(best_dist)
        marginal = marginal_class()
        marginal.fit(data)

        marginal_models[col] = marginal

        # store latest (extreme-informed) marginal
        final_marginal_models[col] = marginal
        final_distribution_summary[col] = best_dist

        print(f"    {col}: {best_dist}")

    # ------------------ Step 4: Transform to uniform space ------------------
    u_data = []

    for col in df_sub.columns:
        u = safe_cdf(marginal_models[col], df_sub[col].values)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        u_data.append(u)

    copula_df_sub = pd.DataFrame(
        np.column_stack(u_data),
        columns=df_sub.columns
    )

    # Store the processed uniform data
    assert not copula_df_sub.empty, "Uniform dataframe is empty"
    all_uniform_data.append(copula_df_sub)

###############################################################################
# Step 5 (FINAL): Fit vine copula using ALL processed subsets
###############################################################################
combined_uniform_df = pd.concat(all_uniform_data, ignore_index=True)





##
# Fit vine copula
# This automatically:
# - selects structure (trees)
# - selects copula families
# - estimates parameters θ
##
import pyvinecopulib as pv

# Fit vine copula (R-vine)
vine = pv.Vinecop(data=combined_uniform_df)


##
# Extract vine structure (graph)
# Output looks like:
# Tree 1: rain — cloud — temp — snow
# Tree 2: conditional dependencies
# ...
##
print(vine.structure)


##
# Extract pair-copulas (θ, τ)
##
for tree_idx, tree in enumerate(vine.pair_copulas):
    print(f"\n--- Tree {tree_idx + 1} ---")

    for edge_idx, cop in enumerate(tree):
        print(f"Edge {edge_idx + 1}:")

        # Copula family
        print(" Family:", cop.family)

        # Parameter θ
        #print(" Parameters (θ):", cop.parameters)

        # Kendall's tau
        tau = cop.tau
        print(" Kendall's τ:", tau)


##
# Compute tail dependence λ
# pyvinecopulib lets you compute this:
##
for tree_idx, tree in enumerate(vine.pair_copulas):
    #print(f"\n--- Tree {tree_idx + 1} Tail Dependence ---")

    for edge_idx, cop in enumerate(tree):
        try:
            lambda_L = cop.lower_tail_dependence()
            lambda_U = cop.upper_tail_dependence()
        except:
            lambda_L, lambda_U = 0, 0

        #print(f"Edge {edge_idx + 1}: λ_L={lambda_L}, λ_U={lambda_U}")





#####################################
# Generate new samples from the vine
#####################################

# Generate new samples
n_new = 100
u_sim = vine.simulate(n_new)

U_sim = pd.DataFrame(u_sim, columns=df.columns)

# Transform back to original scale
df_sim = pd.DataFrame()
for col in df.columns:
    df_sim[col] = np.quantile(df[col], U_sim[col])

# Print to console
print("\n--- Simulated Data (first 10 rows) ---")
print(df_sim.head(10))

# Save to CSV
df_sim.to_csv("simulated_weather_data.csv", index=False)

print("\nCSV file 'simulated_weather_data.csv' has been created.")

