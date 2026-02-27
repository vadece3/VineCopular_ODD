import numpy as np
import pandas as pd
from scipy.stats import gamma, norm, beta, uniform
from copulas.multivariate import VineCopula
from copulas.univariate import GammaUnivariate, GaussianUnivariate, BetaUnivariate, UniformUnivariate

# Step 1: Load your real ODD data
df = pd.read_csv('D:\TU Claustal\CLASSES\Sommer2025\Seminar\Actual_Work\My_work\Test_data\odd_weather_data.csv')  # Replace with your actual file path

# Optional: Check and clean data
df = df[['Rainmm', 'WindSpeedkmPerH', 'TemperaturePercentage', 'CloudCoverPercentage']].dropna()

# Optional: Convert Sun % to fraction
if df['TemperaturePercentage'].max() > 1.0:
    df['TemperaturePercentage'] = df['TemperaturePercentage'] / 100.0

# Step 2: Fit marginal distributions
rain_dist = GammaUnivariate()
rain_dist.fit(df['Rainmm'])

wind_dist = GaussianUnivariate()
wind_dist.fit(df['WindSpeedkmPerH'])

temperature_dist = BetaUnivariate()
temperature_dist.fit(df['TemperaturePercentage'])

cloud_dist = UniformUnivariate()
cloud_dist.fit(df['CloudCoverPercentage'])

#
# print("Rain Beta params:", rain_dist._get_params())
# print("Wind Beta params:", wind_dist._get_params())
# print("Temperature Beta params:", temperature_dist._get_params())
# print("Cloud Beta params:", cloud_dist._get_params())
#

# Step 3: Transform to uniform scale using CDFs
u_rain = rain_dist.cumulative_distribution(df['Rainmm'])
u_wind = wind_dist.cumulative_distribution(df['WindSpeedkmPerH'])
u_temperature = temperature_dist.cumulative_distribution(df['TemperaturePercentage'])
u_cloud = cloud_dist.cumulative_distribution(df['CloudCoverPercentage'])

copula_data = np.column_stack([u_rain, u_wind, u_temperature, u_cloud])
copula_df = pd.DataFrame(copula_data, columns=['Rainmm', 'WindSpeedkmPerH', 'TemperaturePercentage', 'CloudCoverPercentage'])

# Step 4: Fit a vine copula
vine = VineCopula('center')  # C-vine structure
vine.fit(copula_df)

#
# print("Rain:", u_rain.min(), u_rain.max())
# print("Wind:", u_wind.min(), u_wind.max())
# print("Temperature:", u_temperature.min(), u_temperature.max())  # Should be (0, 1)
# print("Cloud:", u_cloud.min(), u_cloud.max())
#
# print(np.isnan(u_rain).sum())  # Should be 0
# print(np.isnan(u_wind).sum())
# print(np.isnan(u_temperature).sum())
# print(np.isnan(u_cloud).sum())
#

# Step 5: Simulate new uniform samples
n_sim = 200  # Number of new samples you want
sim_u = vine.sample(n_sim)


# See the full vine structure
print(vine.trees)

# Remove all rows containing negative values
sim_u = sim_u.clip(0.0001, 0.9999)

# Step 6: Inverse CDF to original scale
sim_rain = rain_dist.percent_point(sim_u.iloc[:, 0])
sim_wind = wind_dist.percent_point(sim_u.iloc[:, 1])
sim_temperature = temperature_dist.percent_point(sim_u.iloc[:, 2])
sim_cloud = cloud_dist.percent_point(sim_u.iloc[:, 3])

# Optional: Reconvert Sun fraction to %
sim_temperature = sim_temperature * 100.0

# Step 7: Create final usable table
simulated_df = pd.DataFrame({
    'Rainmm': sim_rain,
    'WindSpeedkmPerH': sim_wind,
    'TemperaturePercentage': sim_temperature,
    'CloudCoverPercentage': sim_cloud
})

# Remove all negative values
simulated_data = simulated_df[(simulated_df >= 0).all(axis=1)]

# Output preview
print(simulated_data.head())

# Step 8: (Optional) Save to CSV
simulated_data.to_csv('D:\TU Claustal\CLASSES\Sommer2025\Seminar\Actual_Work\My_work\Test_data\simulated_odd_output_result.csv', index=False)



######Extract Pairwise Dependencies from the Vine
labels = ['Rain', 'Wind', 'Sun', 'Cloud']
n = len(labels)
dependency_matrix = np.zeros((n, n))

# Go through vine copula structure and collect dependencies
for tree in vine.trees:
    for edge in tree.edges:
        left = edge.L
        right = edge.R
        copula = edge
        print("Found edge between:", left, "and", right , "and" , copula)
        print("Available labels:", labels)

        # Some copulas (e.g., Gaussian) provide a correlation
        if hasattr(copula, 'tau'):
            tau = copula.tau
        elif hasattr(copula, 'get_dependence'):
            tau = copula.get_dependence()
        else:
            tau = copula.get_dependence()  # Default if unknown

        # Convert to percentage
        strength = round(abs(tau) * 100, 2)

        # Fill symmetric matrix
        i = labels.index(labels[left])
        j = labels.index(labels[right])
        dependency_matrix[i][j] = strength
        dependency_matrix[j][i] = strength  # symmetric

##Now I create a labeled DataFrame to print the result
dependency_df = pd.DataFrame(dependency_matrix, columns=labels, index=labels)
print(dependency_df)
