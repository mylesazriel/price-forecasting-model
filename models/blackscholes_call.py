# Importing libraries
# --------------------------------------------------
import pandas as pd
import numpy as np
import scipy.stats as si

# Black Scholes Call Function
# --------------------------------------------------
sigma = 0.2  # "Just use a proxy for volatility as I have to calculate that first using my methodology."


def black_scholes_call_div(spxclose, strike, ttm, r, q, sigma):

    # spxclose: spot price
    # strike: strike price
    # ttm: time to maturity
    # r: interest rate
    # q: rate of continuous dividend paying asset
    # sigma: volatility of underlying asset

    d1 = (np.log(spxclose / strike) + (r - q + 0.5 *
          sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
    d2 = (np.log(spxclose / strike) + (r - q - 0.5 *
          sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))

    call = (spxclose * np.exp(-q * ttm) * si.norm.cdf(d1, 0.0, 1.0) -
            strike * np.exp(-r * ttm) * si.norm.cdf(d2, 0.0, 1.0))

    print(call)


# Reading and prepping the data
# --------------------------------------------------
data = pd.read_csv('blackscholes_dataset.csv')
data = data[data.ttm != 0]

# Applying the model on every
# row on the dataframe
# --------------------------------------------------
data['call'] = data.apply(lambda row: black_scholes_call_div(
    row['spxclose'], row['strike'], row['ttm'], row['r'], row['q'], sigma), axis=1)
data
