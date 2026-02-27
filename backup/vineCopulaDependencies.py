import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gamma, norm, beta, uniform
from copulas.multivariate import VineCopula
from copulas.univariate import GammaUnivariate, GaussianUnivariate, BetaUnivariate, UniformUnivariate


##########################################################################
#Fucntion to print the dependencies
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
#Print the dependencies
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
#############################################################
##Delete all records which have yeros and take records which clearlz shows complete records, so
#that the test should use all variables to do sampling


##Next powerful extensions (optional)

#If you want, I can help you:

#Force snow only when temperature < 0°C

#Generate winter-only ODDs

#Compare rain-only vs snow-only scenarios

#Add conditional sampling (e.g., “given snow > 5 cm”)
#############################################################


























# ###
# #### Methode 1 (Prints wether it increases or decreases and with the percentages)
# ###

# # Number of samples for approximation
# n_approx = 20
# approx_samples_u = vine.sample(n_approx)
#
# # Unconditional means for all variables (original scale)
# unconditional_means = {}
# for i, var in enumerate(var_names):
#     unconditional_means[var] = df[var].mean()
#
# print("\n🔎 Approximate % effect of 30% increase in each variable on others:")
#
# increase_pct = 0.3
# base_percentile = 0.5
# tolerance = 0.01
#
# for i, var_a in enumerate(var_names):
#     target_percentile = min(base_percentile + increase_pct, 0.9999)
#
#     # Filter samples where var_a's uniform value near target percentile
#     cond_idx = (approx_samples_u.iloc[:, i] >= target_percentile - tolerance) & (approx_samples_u.iloc[:, i] <= target_percentile + tolerance)
#     cond_samples = approx_samples_u.loc[cond_idx]
#
#     if len(cond_samples) == 0:
#         print(f"⚠️ No samples near target percentile for {var_a}. Try increasing tolerance.")
#         continue
#
#     print(f"\nWhen {var_a} increases from median to median + 30% percentile:")
#
#     for j, var_b in enumerate(var_names):
#         if i == j:
#             # Skip self-effect or optionally print 30% increase info
#             continue
#
#         # Get conditional uniform samples of var_b
#         cond_var_u = cond_samples.iloc[:, j]
#
#         # Inverse transform to original scale using corresponding marginal
#         # Note: You must have marginal distributions for each variable like rain_dist, wind_dist...
#         # Store marginals in a list for easy indexing:
#         marginal_list = [rain_dist, wind_dist, temperature_dist, cloud_dist]
#         cond_var_orig = marginal_list[j].percent_point(cond_var_u)
#
#         cond_mean = cond_var_orig.mean()
#         uncond_mean = unconditional_means[var_b]
#
#         pct_change = ((cond_mean - uncond_mean) / uncond_mean) * 100
#
#         sign = "increase" if pct_change > 0 else "decrease"
#
#         print(f" → {var_b}: {abs(pct_change):.2f}% {sign}")
#
#
######################## OR #################################################
#
#
# ###
# #### Methode 2 (Only prints wether it increases or decreases)
# ###
# # Variables and marginals list (adjust if needed)
# var_names = ['Rain_mm', 'Wind_speed_10m_km_per_h', 'Temperature', 'Cloud_cover_percentage']
# marginal_list = [rain_dist, wind_dist, temperature_dist, cloud_dist]
#
# n_approx = 20000
# approx_samples_u = vine.sample(n_approx)
#
# # Calculate unconditional means for reference
# unconditional_means = {var: df[var].mean() for var in var_names}
#
# increase_pct = 0.3
# base_percentile = 0.5
# tolerance = 0.01
#
# print("\nApproximate direction of change in other variables when one variable increases by 30% percentile:")
#
# for i, var_a in enumerate(var_names):
#     target_percentile = min(base_percentile + increase_pct, 0.9999)
#
#     # Filter samples where var_a near target percentile (approx conditioning)
#     cond_idx = (approx_samples_u.iloc[:, i] >= target_percentile - tolerance) & (approx_samples_u.iloc[:, i] <= target_percentile + tolerance)
#     cond_samples = approx_samples_u.loc[cond_idx]
#
#     if len(cond_samples) == 0:
#         print(f"⚠️ No samples near target percentile for {var_a}. Try increasing tolerance.")
#         continue
#
#     print(f"\nWhen {var_a} increases from median to median + 30% percentile:")
#
#     for j, var_b in enumerate(var_names):
#         if i == j:
#             continue  # skip self
#
#         cond_var_u = cond_samples.iloc[:, j]
#         cond_var_orig = marginal_list[j].percent_point(cond_var_u)
#
#         cond_mean = cond_var_orig.mean()
#         uncond_mean = unconditional_means[var_b]
#
#         if cond_mean > uncond_mean:
#             direction = "increase"
#         elif cond_mean < uncond_mean:
#             direction = "decrease"
#         else:
#             direction = "no change"
#
#         print(f" → {var_b}: {direction}")
#


######################## OR #################################################


# ###
# #### Methode 3 (print the increase or decrease in percentage of variable A when we increase a particular percentage in variable B)
# ###
#
# # Your variables and marginals
# var_names = ['Rainmm', 'WindSpeedkmPerH', 'TemperaturePercentage', 'CloudCoverPercentage']
# marginal_list = [rain_dist, wind_dist, temperature_dist, cloud_dist]
#
# n_approx = 30
# approx_samples_u = vine.sample(n_approx)
#
# # Unconditional means for each variable
# unconditional_means = {var: df[var].mean() for var in var_names}
#
# # Define percentiles of variable A to test (e.g., 50% to 90% in steps)
# percentile_steps = np.arange(0.5, 0.95, 0.05)
#
# # Tolerance window for conditioning on percentile (uniform scale)
# tolerance = 0.01
#
# print("\nEstimated effect of % increase in each variable on others:")
#
# for i, var_a in enumerate(var_names):
#     print(f"\nEffects of increasing {var_a}:")
#
#     # Get unconditional mean for var_a to convert uniform percentiles to original scale
#     # We'll calculate % increase relative to median (= 50th percentile)
#     median_orig_a = marginal_list[i].percent_point(0.5)
#
#     for j, var_b in enumerate(var_names):
#         if i == j:
#             continue  # skip self effect
#
#         # Prepare lists for % increase in A and % change in B
#         pct_increases_in_a = []
#         pct_changes_in_b = []
#
#         for p in percentile_steps:
#             # Filter samples near percentile p for var_a
#             cond_idx = (approx_samples_u.iloc[:, i] >= p - tolerance) & (approx_samples_u.iloc[:, i] <= p + tolerance)
#             cond_samples = approx_samples_u.loc[cond_idx]
#             if len(cond_samples) == 0:
#                 continue
#
#             # Calculate original scale means for var_a and var_b in conditional samples
#             cond_a_orig = marginal_list[i].percent_point(cond_samples.iloc[:, i])
#             cond_b_orig = marginal_list[j].percent_point(cond_samples.iloc[:, j])
#
#             mean_a = cond_a_orig.mean()
#             mean_b = cond_b_orig.mean()
#
#             # Calculate % increase in A relative to median
#             pct_increase_a = ((mean_a - median_orig_a) / median_orig_a) * 100
#
#             # Calculate % change in B relative to unconditional mean
#             pct_change_b = ((mean_b - unconditional_means[var_b]) / unconditional_means[var_b]) * 100
#
#             pct_increases_in_a.append(pct_increase_a)
#             pct_changes_in_b.append(pct_change_b)
#
#         # Simple linear regression (fit line: pct_change_b ~ pct_increase_a)
#         if len(pct_increases_in_a) > 1:
#             coef = np.polyfit(pct_increases_in_a, pct_changes_in_b, 1)  # degree 1 polynomial
#             slope = coef[0]
#             intercept = coef[1]
#             direction = "increase" if slope > 0 else "decrease"
#
#             print(
#                 f" - {var_b}: For each 1% increase in {var_a}, {var_b} tends to {direction} by approx {abs(slope):.2f}%")
#         else:
#             print(f" - {var_b}: Not enough data to estimate effect.")

############################################################################################################################

# Step 6: Simulate new uniform samples from copula
n_sim = 200
sim_u = vine.sample(n_sim).clip(0.0001, 0.9999)  # Clip to avoid boundary issues

# Step 7: Inverse CDF to transform back to original scale
sim_rain = rain_dist.percent_point(sim_u.iloc[:, 0])
sim_wind = wind_dist.percent_point(sim_u.iloc[:, 1])
sim_temperature = temperature_dist.percent_point(sim_u.iloc[:, 2])
sim_cloud = cloud_dist.percent_point(sim_u.iloc[:, 3])

# Convert sun/cloud back to percentages
sim_temperature *= 100.0
sim_cloud *= 100.0

# Step 8: Create final DataFrame
simulated_df = pd.DataFrame({
    'Rainmm': sim_rain,
    'WindSpeedkmPerH': sim_wind,
    'TemperaturePercentage': sim_temperature,
    'CloudCoverPercentage': sim_cloud
})

# Remove negative values (if any)
simulated_df = simulated_df[(simulated_df >= 0).all(axis=1)]

# Step 9: Save simulated data to CSV
output_path = r'D:\TU Claustal\CLASSES\Sommer2025\Seminar\Actual_Work\My_work\Test_data\simulated_odd_output_result.csv'
simulated_df.to_csv(output_path, index=False)

# Output preview on console
# print(simulated_df.head())






# # Step 10: Optional - Plot distributions for validation
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# variables = ['Rainmm', 'WindSpeedkmPerH', 'TemperaturePercentage', 'CloudCoverPercentage']
#
# for i, var in enumerate(variables):
#     ax = axes[i // 2, i % 2]
#     sns.kdeplot(df[var.split('_')[0]], label='Original', ax=ax)
#     sns.kdeplot(simulated_df[var], label='Simulated', ax=ax)
#     ax.set_title(f'{var} Distribution')
#     ax.legend()
#
# plt.tight_layout()
# plt.show()


# Step 9: Analyze directional dependencies
# var_names = ['Rainmm', 'WindSpeedkmPerH', 'TemperaturePercentage', 'CloudCoverPercentage']
# marginals = [rain_dist, wind_dist, temperature_dist, cloud_dist]
# real_means = {var: df[var].mean() for var in var_names}
#
# # Resample from the copula for dependency analysis
# n_approx = 30
# sampled_u = vine.sample(n_approx)
#
# # Thresholds and tolerance to simulate increases
# percentiles = np.arange(0.5, 0.95, 0.05)
# tolerance = 0.01
#
# print("\nEstimated direction of dependency (increase/decrease):\n")
#
# for i, var_a in enumerate(var_names):
#     print(f"\n↑ {var_a} causes:")
#     median_a = marginals[i].percent_point(0.5)
#
#     for j, var_b in enumerate(var_names):
#         if i == j:
#             continue
#
#         pct_increase_a = []
#         pct_change_b = []
#
#         for p in percentiles:
#             # Conditional subsample where variable A is "around" percentile p
#             mask = (
#                 (sampled_u.iloc[:, i] >= p - tolerance) &
#                 (sampled_u.iloc[:, i] <= p + tolerance)
#             )
#             cond_sample = sampled_u[mask]
#
#             if len(cond_sample) < 50:
#                 continue
#
#             a_vals = marginals[i].percent_point(cond_sample.iloc[:, i])
#             b_vals = marginals[j].percent_point(cond_sample.iloc[:, j])
#
#             mean_a = np.mean(a_vals)
#             mean_b = np.mean(b_vals)
#
#             delta_a = (mean_a - median_a) / median_a * 100
#             delta_b = (mean_b - real_means[var_b]) / real_means[var_b] * 100
#
#             pct_increase_a.append(delta_a)
#             pct_change_b.append(delta_b)
#
#         # Linear regression (approximate slope of effect)
#         if len(pct_increase_a) > 1:
#             slope = np.polyfit(pct_increase_a, pct_change_b, 1)[0]
#             direction = "increase" if slope > 0 else "decrease"
#             print(f"   → {var_b}: tends to {direction} by ~{abs(slope):.2f}% per 1% increase")
#         else:
#             print(f"   → {var_b}: insufficient data to estimate")