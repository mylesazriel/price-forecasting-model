import pandas as pd
import numpy as np
import scipy.stats as si


def black_scholes_call_div(S, K, T, r, q, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # q: rate of continuous dividend paying asset
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) -
            K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call


### Reading and formatting .csv
data = pd.read_csv('blackscholes_dataset.csv')
data = data[data.ttm != 0]
data.head()

# Trying to figure out how to step the function
# through every row in the dataset
# Try the .apply() function?
S_list = data['spxclose'].values.tolist()
K_list = data['strike'].values.tolist()
T_list = data['ttm'].values.tolist()
r_list = data['r'].values.tolist()
q_list = data['q'].values.tolist()
sigma = 0.2

S = S_list[0]
K = K_list[0]
T = T_list[0]
r = r_list[0]
q = q_list[0]

print(S)

for i in S_list:
    S_list.pop(0)
    print(S)
