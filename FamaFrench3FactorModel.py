import pandas as pd 
import yfinance as yf 
import statsmodels.api as sm 
import getFamaFrenchFactors as gff 

class FamaFrench3FactorModel:

	def __init__(self):
		self.numFactors = 3 

	def getPrediction(self, ticker, start, end):
		self.ticker = ticker 
		self.start = start 
		self.end = end 

		#Retrieve price data for the stock
		stock_data = self.getPrices(ticker, start, end)

		#Retrieve the Fama French benchmark data 
		#Returned benchmark data are on a monthly basis 
		#Rename the "Date" column and set it as an index of the dataframe 
		ff3_monthly = gff.famaFrench3Factor(frequency='m')
		ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
		ff3_monthly.set_index('Date', inplace=True)

		#Calculate historical monthly returns of the stock 
		stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
		stock_returns.name = "Month_Rtn"
		ff_data = ff3_monthly.merge(stock_returns, on='Date')

		#Calculate beta 
		X = ff_data[['Mkt-RF', 'SMB', 'HML']]
		#Calculate stock premium by subtracting risk free rate
		y = ff_data['Month_Rtn'] - ff_data['RF']
		X = sm.add_constant(X)

		#Use stats model to calculate betas
		ff_model = sm.OLS(y, X).fit()

		intercept, b1, b2, b3 = ff_model.params

		#Calculate estimation of expected returns of stock
		rf = ff_data['RF'].mean()
		market_premium = ff3_monthly['Mkt-RF'].mean()
		size_premium = ff3_monthly['SMB'].mean()
		value_premium = ff3_monthly['HML'].mean()

		expected_monthly_return = rf + (b1 * market_premium) + (b2 * size_premium) + (b3 * value_premium)
		expected_yearly_return = expected_monthly_return * 12

		#return both expected monthly return and expected yearly return and upload both to firebase 
		return expected_monthly_return, expected_yearly_return


	def getPrices(self, ticker, start, end):
		stock_data = yf.download(ticker, start, end)
		return stock_data

#run in main to make sure it works


