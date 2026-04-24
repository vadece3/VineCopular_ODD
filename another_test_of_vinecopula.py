import numpy as np
import pandas as pd
from scipy import stats
from copulas.bivariate import Clayton

##
# Generate sample dependent data
# Right now:
# - Data is dependent
# - But not in copula form yet
##
np.random.seed(42)


# Generate correlated normal data
mean = [0, 0]
cov = [[1, 0.7], [0.7, 1]]  # correlation = 0.7

data = np.random.multivariate_normal(mean, cov, size=1000)
x = data[:, 0]
y = data[:, 1]


##
# Transform to uniform
# Now:
# - u, v ∈ (0,1)
# - This is the copula input
##
u = stats.rankdata(x) / (len(x) + 1)
v = stats.rankdata(y) / (len(y) + 1)

U = pd.DataFrame({'u': u, 'v': v})


##
# Fit a copula → get θ
# We use a Clayton copula (good for lower tail dependence):
# Output:
# - θ = dependency parameter
# - Larger θ → stronger dependence
##
copula = Clayton()
copula.fit(U.values)

theta = copula.theta
print("Theta (θ):", theta)

##
# Optional: Try another copula (Gaussian)
# For Gaussian:
# - No tail dependence:
# - λ-lower = λ-upper = 0
##
#        from copulas.bivariate import GaussianCopula
#        copula = GaussianCopula()
#        copula.fit(U.values)
#        rho = copula.rho
#        print("Correlation (ρ):", rho)


##
# Compute Kendall’s τ
##

##
# Method 1 (direct from data)
##
tau, _ = stats.kendalltau(x, y)
print("Kendall's tau (τ):", tau)

##
# Method 2 (from copula parameter θ)
# For Clayton copula, the relationship is:
#
# 𝜏 = 𝜃 / (𝜃 + 2)
##
tau_from_theta = theta / (theta + 2)
print("Tau from θ:", tau_from_theta)


##
# Compute tail dependence λ
# For Clayton copula:
#
# Lower tail dependence:
# λ-lower = 2 TO-THE-POWER(−1/θ)
#
# Upper tail dependence:
# 𝜆-upper = 0
##
lambda_L = 2 ** (-1 / theta)
lambda_U = 0

print("Lower tail dependence (λ_L):", lambda_L)
print("Upper tail dependence (λ_U):", lambda_U)


##
# Final output summary
##
print("\n-----Final output summary-----")
print("\n--- Dependency Summary ---")
print(f"Theta (θ): {theta}")
print(f"Kendall's tau (τ): {tau}")
print(f"Tau from θ: {tau_from_theta}")
print(f"Lower tail dependence (λ_L): {lambda_L}")
print(f"Upper tail dependence (λ_U): {lambda_U}")
