import numpy as np
import pandas as pd

##
# Generate realistic weather-like data
##
np.random.seed(42)
n = 1000

# Latent correlation structure
corr = np.array([
    [1.0,  0.7, -0.5,  0.6],  # rain
    [0.7,  1.0, -0.6,  0.5],  # cloud
    [-0.5, -0.6, 1.0, -0.4],  # temperature
    [0.6,  0.5, -0.4, 1.0]    # snow
])

# Generate multivariate normal
data = np.random.multivariate_normal(mean=[0,0,0,0], cov=corr, size=n)

df = pd.DataFrame(data, columns=["rain", "cloud", "temp", "snow"])


##
# Transform to uniform marginals (copula input)
# Now:
# - All variables are in (0,1)
# - Ready for copula fitting
##
from scipy.stats import rankdata

U = df.copy()
for col in U.columns:
    U[col] = rankdata(U[col]) / (n + 1)


##
# Fit vine copula
# This automatically:
# - selects structure (trees)
# - selects copula families
# - estimates parameters θ
##
import pyvinecopulib as pv

# Convert to numpy
u_data = U.to_numpy()

# Fit vine copula (R-vine)
vine = pv.Vinecop(data=u_data)


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
        print(" Parameters (θ):", cop.parameters)

        # Kendall's tau
        tau = cop.tau
        print(" Kendall's τ:", tau)


##
# Compute tail dependence λ
# pyvinecopulib lets you compute this:
##
for tree_idx, tree in enumerate(vine.pair_copulas):
    print(f"\n--- Tree {tree_idx + 1} Tail Dependence ---")

    for edge_idx, cop in enumerate(tree):
        try:
            lambda_L = cop.lower_tail_dependence()
            lambda_U = cop.upper_tail_dependence()
        except:
            lambda_L, lambda_U = 0, 0

        print(f"Edge {edge_idx + 1}: λ_L={lambda_L}, λ_U={lambda_U}")


##
# Example output (interpretation)
# You’ll get something like:
# Tree 1:
# Rain — Cloud: Gaussian, ρ=0.68, τ=0.48
# Cloud — Temp: Clayton, θ=1.8, τ=0.47, λ_L=0.3
# Rain — Snow: Gumbel, θ=2.1, τ=0.52, λ_U=0.4

##
# What this means (IMPORTANT)
# ✔ Structural representation
# Vine = sequence of trees
# Edges = dependencies
##

##
# Parameter representation
# Each edge has:
# - copula family
# - θ (or ρ)
#
# Rank correlation
# - τ per edge → strength of dependency
#
# Tail dependence
# - λ_L → joint extreme low events
# - λ_U → joint extreme high events
#
#
# Final takeaway
#
# After vine copula fitting, your dependency is represented as:
#
#  Graph
#  - Tree structure (who depends on whom)
#
#  Pair-copulas
#  Each edge has:
#  - Family (Gaussian, Clayton, etc.)
#  - θ (parameter)
#  - τ (Kendall’s tau)
#  - λ (tail dependence)
##
