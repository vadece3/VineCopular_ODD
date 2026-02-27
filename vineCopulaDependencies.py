import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from copulas.multivariate import VineCopula
from copulas.univariate import (
    GammaUnivariate,
    GaussianUnivariate,
    BetaUnivariate,
    UniformUnivariate
)

###############################################################################
# Helper: Print vine dependencies
###############################################################################
def describe_vine_dependencies(vine_model, data_df, var_names):
    print("📊 Pairwise Dependency Ratios from Vine Copula:\n")

    # Precompute marginal variability in uniform space
    marginal_std = {
        col: np.std(data_df[col].values)
        for col in data_df.columns
    }

    for tree_idx, tree in enumerate(vine_model.trees):
        print(f"\n🌲 Tree Level {tree_idx + 1}:")

        for edge in tree.edges:
            try:
                i = edge.L
                j = edge.R

                var_i = var_names[i]
                var_j = var_names[j]

                tau = getattr(edge, 'kendall_tau', None)
                if tau is None:
                    tau = getattr(edge, 'tau', None)

                if tau is None:
                    print(f" ⚠️ No Kendall’s τ for {var_i} ~ {var_j}")
                    continue

                tau_abs = abs(tau)

                # Relative contribution weights
                w_i = marginal_std[var_i]
                w_j = marginal_std[var_j]

                total = w_i + w_j
                if total == 0:
                    continue

                pct_i = int((w_i / total) * tau_abs * 100)
                pct_j = int((w_j / total) * tau_abs * 100)

                print(
                    f" - {var_i} ~ {var_j}: → "
                    f"{pct_i}% : {pct_j}% dependency"
                )

            except Exception as e:
                print(f" ⚠️ Error processing edge: {e}")

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

print(f"\n📐 Extreme selection: n = {n} rows per column\n")

subset_dfs = {}

for col in df.columns:
    subset_dfs[col] = (
        df.sort_values(by=col, ascending=False)
          .head(n)
          .reset_index(drop=True)
    )

    print(f" ✔ Subset created for column: {col}")


###############################################################################
# Steps 2–5 (REPEATED per column subset)
###############################################################################
all_uniform_data = []

final_marginal_models = {}
final_distribution_summary = {}

for driving_col, sub_df in subset_dfs.items():

    print(f"\n🔁 Processing subset driven by: {driving_col}")

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
    print(" 📦 Marginal Distribution Selection:")

    for col in df_sub.columns:
        data = df_sub[col].values
        best_dist = detect_best_distribution(data)

        marginal_class = COPULA_MARGINAL_MAP.get(best_dist, GaussianUnivariate)
        marginal = marginal_class()
        marginal.fit(data)

        marginal_models[col] = marginal

        # store latest (extreme-informed) marginal
        final_marginal_models[col] = marginal
        final_distribution_summary[col] = best_dist

        print(f"   ✔ {col}: {best_dist}")

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

vine = VineCopula('regular')
vine.fit(combined_uniform_df)

# ###############################################################################
# # Step 2: Preprocessing
# ###############################################################################
# if df['CloudCoverPercentage'].max() > 1.0:
#     df['CloudCoverPercentage'] /= 100.0
#
# df['Rainmm'] = df['Rainmm'].clip(lower=0.001)
# df['Snowfallcm'] = df['Snowfallcm'].clip(lower=0.001)
#
# ###############################################################################
# # Step 3: Automatic marginal fitting
# ###############################################################################
# marginal_models = {}
# distribution_summary = {}
#
# print("\n📦 Marginal Distribution Selection:\n")
#
# for col in df.columns:
#     data = df[col].values
#
#     best_dist = detect_best_distribution(data)
#     if best_dist is None:
#         raise ValueError(f"No valid distribution found for {col}")
#
#     marginal_class = COPULA_MARGINAL_MAP.get(best_dist, GaussianUnivariate)
#     marginal = marginal_class()
#     marginal.fit(data)
#
#     marginal_models[col] = marginal
#     distribution_summary[col] = best_dist
#
#     print(f" ✔ {col}: {best_dist} → {marginal_class.__name__}")
#
# ###############################################################################
# # Step 4: Transform to uniform space
# ###############################################################################
# u_data = []
# for col in df.columns:
#     u = marginal_models[col].cumulative_distribution(df[col])
#     u = np.clip(u, 1e-6, 1 - 1e-6)
#     u_data.append(u)
#
# copula_df = pd.DataFrame(
#     np.column_stack(u_data),
#     columns=df.columns
# )
#
# assert copula_df.shape[1] == len(df.columns)
#
# ###############################################################################
# # Step 5: Fit vine copula
# ###############################################################################
# vine = VineCopula('regular')
# vine.fit(copula_df)

###############################################################################
# Dependency report
###############################################################################
describe_vine_dependencies(
    vine_model=vine,
    data_df=combined_uniform_df,
    var_names=list(combined_uniform_df.columns)
)

###############################################################################
# Step 6: Sample from vine copula
###############################################################################
n_samples = 300
sim_u = vine.sample(n_samples)
sim_u = sim_u.clip(1e-4, 1 - 1e-4)

###############################################################################
# Step 7: Inverse CDF to original scale
###############################################################################
sim_data = {}

for i, col in enumerate(combined_uniform_df.columns):

    # --- Inverse CDF ---
    values = safe_ppf(
        final_marginal_models[col],
        sim_u.iloc[:, i].values
    )

    # --- Physical constraints per variable ---
    if col == 'Rainmm':
        values = np.maximum(values, 0.0)
        values[values == 0.0] = 0.001

    elif col == 'WindSpeedkmPerH':
        values = np.maximum(values, 0.0)

    elif col == 'TemperatureDegreeCelcius':
        # No constraints
        pass

    elif col == 'CloudCoverPercentage':
        values = np.clip(values, 0.0, 1.0) * 100.0

    elif col == 'Snowfallcm':
        values = np.maximum(values, 0.0)
        values[values == 0.0] = 0.001

    # --- Store result ---
    sim_data[col] = values

# Final simulated dataframe
predicted_df = pd.DataFrame(sim_data)

# ###############################################################################
# # Step 8: Physical constraints
# ###############################################################################
# # predicted_df = predicted_df[
# #     (predicted_df['Rainmm'] >= 0) &
# #     (predicted_df['WindSpeedkmPerH'] >= 0) &
# #     (predicted_df['CloudCoverPercentage'] >= 0) &
# #     (predicted_df['Snowfallcm'] >= 0)
# # ]
# #
# # predicted_df = predicted_df.round(1)
#
# # --- Hard physical bounds ---
# predicted_df = predicted_df[
#     (predicted_df['Rainmm'] >= 0.0) &
#     (predicted_df['WindSpeedkmPerH'] >= 0.0) &
#     (predicted_df['CloudCoverPercentage'] >= 0.0) &
#     (predicted_df['CloudCoverPercentage'] <= 100.0) &
#     (predicted_df['Snowfallcm'] >= 0.0)
# ]
#
# # --- Optional physical coherence rules ---
# # Snowfall should not occur at very high temperatures
# predicted_df.loc[
#     predicted_df['TemperatureDegreeCelcius'] > 5,
#     'Snowfallcm'
# ] = 0.0
#
# # --- Numerical cleanup ---
# predicted_df = predicted_df.replace([np.inf, -np.inf], np.nan)
# predicted_df = predicted_df.dropna()
#
# # --- Rounding to measurement resolution ---
# predicted_df = predicted_df.round({
#     'Rainmm': 1,                     # mm
#     'WindSpeedkmPerH': 1,             # km/h
#     'TemperatureDegreeCelcius': 1,    # °C
#     'CloudCoverPercentage': 0,        # %
#     'Snowfallcm': 1                  # cm
# })

###############################################################################
# Step 8: Physical + meteorological coherence rules
###############################################################################

# ---------------- Basic physical bounds ----------------
predicted_df = predicted_df[
    (predicted_df['Rainmm'] >= 0.0) &
    (predicted_df['WindSpeedkmPerH'] >= 0.0) &
    (predicted_df['CloudCoverPercentage'] >= 0.0) &
    (predicted_df['CloudCoverPercentage'] <= 100.0) &
    (predicted_df['Snowfallcm'] >= 0.0)
]

# ---------------- Weather logic parameters ----------------
MIN_CLOUD_FOR_PRECIP = 40.0    # %
SNOW_TEMP_MAX = 2.0            # °C (snow above this is unlikely)
RAIN_TEMP_MIN = -2.0           # °C (rain below this is unlikely)

# ---------------- Precipitation requires clouds ----------------
low_cloud = predicted_df['CloudCoverPercentage'] < MIN_CLOUD_FOR_PRECIP

predicted_df.loc[low_cloud, 'Rainmm'] = 0.0
predicted_df.loc[low_cloud, 'Snowfallcm'] = 0.0

# ---------------- Temperature phase logic ----------------
# Warm → no snow
predicted_df.loc[
    predicted_df['TemperatureDegreeCelcius'] > SNOW_TEMP_MAX,
    'Snowfallcm'
] = 0.0

# Cold → suppress rain
predicted_df.loc[
    predicted_df['TemperatureDegreeCelcius'] < RAIN_TEMP_MIN,
    'Rainmm'
] = 0.0

# ---------------- Rain–snow mutual exclusivity ----------------
both_precip = (
    (predicted_df['Rainmm'] > 0) &
    (predicted_df['Snowfallcm'] > 0)
)

# Keep the dominant phase
predicted_df.loc[both_precip &
                 (predicted_df['TemperatureDegreeCelcius'] <= 0),
                 'Rainmm'] = 0.0

predicted_df.loc[both_precip &
                 (predicted_df['TemperatureDegreeCelcius'] > 0),
                 'Snowfallcm'] = 0.0

# ---------------- Numerical cleanup ----------------
predicted_df = predicted_df.replace([np.inf, -np.inf], np.nan)
predicted_df = predicted_df.dropna()

# ---------------- Measurement resolution ----------------
predicted_df = predicted_df.round({
    'Rainmm': 1,
    'WindSpeedkmPerH': 1,
    'TemperatureDegreeCelcius': 1,
    'CloudCoverPercentage': 0,
    'Snowfallcm': 1
})

###############################################################################
# Step 9: Output
###############################################################################
print("\n📈 Predicted ODD Scenarios (First 10 Rows):\n")
print(predicted_df.head(10))

###############################################################################
# Step 10: Save to CSV
###############################################################################
predicted_df.to_csv(output_path,
    index=False
)