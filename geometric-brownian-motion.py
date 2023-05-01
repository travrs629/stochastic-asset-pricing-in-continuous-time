""" Asset Pricing in continuous time with the Geometric Brownian Motion (GBM) and the Monte-Carlo simulation"""

"""
Author: Matteo Bottacini, matteo.bottacini@usi.ch
Last update: March 4, 2021
"""
## GBM is a mathematical model used to describe the evolution of an asset price over time.
## Monte-Carlo simulation is a statistic technique used to simulate the behaviour of a system over time.

# import modules
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# main variables
# stock_name    :   ticker symbol from yahoo finance
# start_date    :   start date to download prices
# end_date      :   end date to download prices
# pred_end_date :   date until which you want to predict price
# scen_size     :   different possible scenarios
stock_name = 'AAPL'
start_date = '2010-01-01'
end_date = '2020-10-31'
pred_end_date = '2020-12-31'
scen_size = 10000

# download and prepare data
## Download daily adjusted closing of a given stock using the Yahoo Finance (yf) API.
## Remember it is adjusted, would it be better if not adjusted?
prices = yf.download(tickers=stock_name, start=start_date, end=pred_end_date)['Adj Close']
train_set = prices.loc[:end_date]
test_set = prices.loc[end_date:pred_end_date]
## Calculates the daily returns of the training set using the closing prices.
daily_returns = ((train_set / train_set.shift(1)) - 1)[1:]

# Geometric Brownian Motion (GBM)

# Parameter Definitions

# So    :   initial stock price
# dt    :   time increment -> a day in our case
# T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
# N     :   number of time points in prediction the time horizon -> T/dt
# t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
# mu    :   mean of historical daily returns
# sigma :   standard deviation of historical daily returns
# b     :   array for brownian increments
# W     :   array for brownian path


# Parameter Assignments
## The last available price in the training set is taken as the initial stock price for the simulation.
So = train_set[-1]
## The time step for the simulation is set to 1 day.
dt = 1  # day   # User input
## The number of weekdays between the end date of the training set and the predicted end date.
## Calculated using date_range() function to generate a series of data, then using map() and lambda() to check if it is weekdays.
## sum() function is used to count the number of weekdays in total.
n_of_wkdays = pd.date_range(start=pd.to_datetime(end_date,
                                                 format="%Y-%m-%d") + pd.Timedelta('1 days'),
                            end=pd.to_datetime(pred_end_date,
                                               format="%Y-%m-%d")).to_series().map(lambda x: 1 if x.isoweekday() in range(1, 6) else 0).sum()
T = n_of_wkdays
## Total number of time steps for the simulation (using T/dt).
N = T / dt
## An array of integers representing the time steps from 1 to 'N'.
t = np.arange(1, int(N) + 1)
## The mean daily return of the stock, calculated using the mean() function.
mu = np.mean(daily_returns)
## The standard deviation of the daily returns of the stock, calculated using std() funciton.
sigma = np.std(daily_returns)
## A dictionary of random numbers generated using the normal distribution with a meano f 0 and a standard deviation of 1.
## It has scen_size numbers of scenarios, each with 'N' number of random numbers.
b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
## A dictionary of cumulative sums of the random numbers in 'b' for each scenario.
W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}


# Calculating drift and diffusion components
## The drift component represents the expected change in the stock price over time.
drift = (mu - 0.5 * sigma ** 2) * t
## The diffusion component represents the random, unpredictable fluctuations in the stock price.
diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}

# Making the predictions
## The stock price simulations are made using the formula below.
S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)])
S = np.hstack((np.array([[So] for scen in range(scen_size)]), S))  # add So to the beginning series
S_max = [S[:, i].max() for i in range(0, int(N))]
S_min = [S[:, i].min() for i in range(0, int(N))]
S_pred = .5 * np.array(S_max) + .5 * np.array(S_min)
final_df = pd.DataFrame(data=[test_set.reset_index()['Adj Close'], S_pred],
                        index=['real', 'pred']).T
final_df.index = test_set.index
mse = 1/len(final_df) * np.sum((final_df['pred'] - final_df['real']) ** 2)

# Plotting the simulations
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('Monte-Carlo Simulation: ' + str(scen_size) + ' simulations', fontsize=20)
plt.title('Asset considered: {}'.format(stock_name))
plt.ylabel('USD Price')
plt.xlabel('Prediction Days')
for i in range(scen_size):
    plt.plot(pd.date_range(start=train_set.index[-1],
                           end=pred_end_date,
                           freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna(), S[i, :])
plt.show()

# Plotting the final prediction against the real price
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.suptitle('Predicted Price vs Real Price', fontsize=20)
plt.title('Mean Squared Error (MSE): {}'.format(np.round(mse, 2)))
plt.ylabel('USD Price')
plt.plot(final_df)
plt.legend(['Real Price', 'Predicted Price'],
           loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
plt.show()
