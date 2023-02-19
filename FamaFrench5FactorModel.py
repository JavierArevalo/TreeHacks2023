import pandas as pd 
import yfinance as yf 
import statsmodels.api as sm 
import getFamaFrenchFactors as gff 

class FamaFrench5FactorModel:

	def __init__(self):
		self.numFactors = 5 

	def getPrediction(self, ticker, start, end):
		self.ticker = ticker 
		self.start = start 
		self.end = end 

		#Retrieve price data for the stock
		stock_data = self.getPrices(ticker, start, end)

		#Retrieve the Fama French benchmark data 
		ff5_monthly = gff.famaFrench5Factor(frequency='m')
		ff5_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
		ff5_monthly.set_index('Date', inplace=True)

		#Calculate historical monthly returns of the stock 
		stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
		stock_returns.name = "Month_Rtn"
		ff_data = ff5_monthly.merge(stock_returns, on='Date')

		#Calculate beta 
		X = ff_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
		#Calculate stock premium by subtracting risk free rate 
		y = ff_data['Month_Rtn'] - ff_data['RF']
		X = sm.add_constant(X)

		#Calculate Beta using stats model OLS 
		ff_model = sm.OLS(y, X).fit()
		intercept, b1, b2, b3, b4, b5 = ff_model.params 

		#Calculate estimation of expected returns of stock 
		rf = ff_data['RF'].mean()
		market_premium = ff5_monthly['Mkt-RF'].mean()
		size_premium = ff5_monthly['SMB'].mean()
		value_premium = ff5_monthly['HML'].mean()
		profit_premium = ff5_monthly['RMW'].mean()
		investing_premium = ff5_monthly['CMA'].mean()

		expected_monthly_return = rf + (b1 * market_premium) + (b2 * size_premium) + (b3 * value_premium) + (b4 * profit_premium) + (b5 * investing_premium)
		expected_yearly_return = expected_monthly_return * 12

		#Return both expected monthly return and expected yearly return and save both in firebase
		return expected_monthly_return, expected_yearly_return


