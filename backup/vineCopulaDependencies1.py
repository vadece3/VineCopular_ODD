import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from copulas.multivariate import VineCopula
from copulas.univariate import GammaUnivariate, GaussianUnivariate, BetaUnivariate, UniformUnivariate



# ---------------------------------------
# Distribution detection logic
# ---------------------------------------

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
        return None, None

    results.sort(key=lambda x: x[1])  # lowest AIC wins
    return results[0]  # (name, aic, params)

# ---------------------------------------
# Map detected distributions to copulas
# ---------------------------------------

COPULA_MARGINAL_MAP = {
    "normal": GaussianUnivariate,
    "gamma": GammaUnivariate,
    "beta": BetaUnivariate,
    "exponential": GammaUnivariate,   # exponential ⊂ gamma
    "lognormal": GaussianUnivariate,  # fallback
    "weibull": GammaUnivariate,       # fallback
    "student_t": GaussianUnivariate   # fallback
}

##########################################################################
#Main function to print the dependencies
def describe_vine_dependencies(vine_model, var_names):
    print("📊 Pairwise Dependencies from Vine Copula:\n")

    for tree_idx, tree in enumerate(vine_model.trees):
        print(f"\n🌲 Tree Level {tree_idx + 1}:")

        for edge in tree.edges:
            try:
                left = edge.L
                right = edge.R

                left_var = var_names[left]
                right_var = var_names[right]

                # Some versions use `tau` or `kendall_tau`, try both
                #Kendall’s tau (τ): a rank correlation coefficient commonly used in copulas
                tau = getattr(edge, 'kendall_tau', None)
                if tau is None:
                    tau = getattr(edge, 'tau', None)

                if tau is None:
                    print(f" ⚠️  No Kendall's tau found for: {left_var} ~ {right_var}")
                    continue

                strength_pct = int(abs(tau) * 100)
                print(f" - {left_var} ~ {right_var}: τ = {tau:.2f} → ~{strength_pct}% dependency")
            # print(f" - {right_var} depends on {left_var} with a Dependency-Percentage of {strength_pct}% ")

            except Exception as e:
                print(f" ⚠️ Error processing edge: {e}")

##################################################################################################################


# Step 1: Load your real ODD data
file_path = r'D:\TU Claustal\CLASSES\Sommer2025\Seminar\Actual_Work\My_work\Test_data\open_meteo_51.78N10.35E563m.csv'
df = pd.read_csv(file_path)

# Keep relevant columns and drop missing values
df = df[['Rainmm', 'WindSpeedkmPerH', 'TemperatureDegreeCelcius', 'CloudCoverPercentage', 'Snowfallcm']].dropna()

# Step 2: Rescale percentage-based variables
# Convert Cloud Cover % to fraction for Uniform fitting
if df['CloudCoverPercentage'].max() > 1.0:
    df['CloudCoverPercentage'] /= 100.0

df['Rainmm'] = df['Rainmm'].clip(lower=0.001)
df['Snowfallcm'] = df['Snowfallcm'].clip(lower=0.001)


# Step 3: Fit marginal distributions
rain_dist = GammaUnivariate()
rain_dist.fit(df['Rainmm'])

wind_dist = GaussianUnivariate()
wind_dist.fit(df['WindSpeedkmPerH'])

temperature_dist = GaussianUnivariate()
temperature_dist.fit(df['TemperatureDegreeCelcius'])

cloud_dist = BetaUnivariate()
cloud_dist.fit(df['CloudCoverPercentage'])

snow_dist = GammaUnivariate()
snow_dist.fit(df['Snowfallcm'])


# Step 4: Transform data to uniform scale using CDFs
u_rain = rain_dist.cumulative_distribution(df['Rainmm'])
u_wind = wind_dist.cumulative_distribution(df['WindSpeedkmPerH'])
u_temperature = temperature_dist.cumulative_distribution(df['TemperatureDegreeCelcius'])
u_cloud = cloud_dist.cumulative_distribution(df['CloudCoverPercentage'])
u_snow = snow_dist.cumulative_distribution(df['Snowfallcm'])

#This prevents boundary failures inside the vine
EPS = 1e-6
u_rain = np.clip(u_rain, EPS, 1 - EPS)
u_wind = np.clip(u_wind, EPS, 1 - EPS)
u_temperature = np.clip(u_temperature, EPS, 1 - EPS)
u_cloud = np.clip(u_cloud, EPS, 1 - EPS)
u_snow = np.clip(u_snow, EPS, 1 - EPS)



copula_data = np.column_stack([u_rain, u_wind, u_temperature, u_cloud, u_snow])
copula_df = pd.DataFrame(copula_data, columns=['Rainmm', 'WindSpeedkmPerH', 'TemperatureDegreeCelcius', 'CloudCoverPercentage', 'Snowfallcm'])

#Sanity check before sampling: This prevents silent errors (silent errors: not considering a variable whose values are all 0 or empty).
assert copula_df.shape[1] == 5

# Step 5: Fit a vine copula
#vine = VineCopula('center')  # C-vine
#vine = VineCopula('regular')  # D-vine
vine = VineCopula('regular')  # R-vine

vine.fit(copula_df)


############################################################################################################################
#Print the dependencies in console
var_names = ['Rainmm', 'WindSpeedkmPerH', 'TemperatureDegreeCelcius', 'CloudCoverPercentage', 'Snowfallcm']
describe_vine_dependencies(vine, var_names)
##################################################################################################################

###########
# Now generate predicted data
############

# Step 6: Generate new predicted samples (uniform space)
n_samples = 300  # number of predicted ODD scenarios
sim_u = vine.sample(n_samples)

# Safety: ensure values are strictly inside (0,1)
sim_u = sim_u.clip(1e-4, 1 - 1e-4)

# Step 7: Inverse CDF to original scale
sim_rain = rain_dist.percent_point(sim_u.iloc[:, 0])
sim_wind = wind_dist.percent_point(sim_u.iloc[:, 1])
sim_temperature = temperature_dist.percent_point(sim_u.iloc[:, 2])
sim_cloud = cloud_dist.percent_point(sim_u.iloc[:, 3])
sim_snow = snow_dist.percent_point(sim_u.iloc[:, 4])

#Cloud is currently in fraction → convert back to %
sim_cloud = sim_cloud * 100.0

#Step 8: Create final predicted dataset
predicted_df = pd.DataFrame({
    'Rainmm': sim_rain,
    'WindSpeedkmPerH': sim_wind,
    'TemperatureDegreeCelcius': sim_temperature,
    'CloudCoverPercentage': sim_cloud,
    'Snowfallcm': sim_snow
})

# Allow negative temperature, restrict others
predicted_df = predicted_df[
    (predicted_df['Rainmm'] >= 0) &
    (predicted_df['WindSpeedkmPerH'] >= 0) &
    (predicted_df['CloudCoverPercentage'] >= 0) &
    (predicted_df['Snowfallcm'] >= 0)
]

# Round values
predicted_df = predicted_df.round(1)


#Step 9: Print the predicted data
print("\n📈 Predicted ODD Scenarios (First 10 Rows):\n")
print(predicted_df.head(10))


#Step 10: Save to CSV
predicted_df.to_csv(
    r'D:\TU Claustal\CLASSES\Sommer2025\Seminar\Actual_Work\My_work\Test_data\predicted_odd_samples.csv',
    index=False
)

