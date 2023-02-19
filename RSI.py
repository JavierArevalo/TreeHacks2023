import pandas as pd 
import talib # Technical Indicator Library 

# RSI Class: create an instance of this class by giving the parameters:
# Buy Threshold Parameter: the minimum value RSI should take to give a buy signal. e.g. 30
# Sell Threshold Parameter: maximum value RSI should take to give a sell signal e.g. 70


class RSI:

	def __init__(self, buyThresholdParam, sellThresholdParam, df):
		self.buyThresholdParam = buyThresholdParam
		self.sellThresholdParam = sellThresholdParam
		self.df = df.copy(deep=False)

	# Create a Standard RSI DataFrame for 5, 10, 14, 30, 90, 180 days 
	def createRSI(self):
		RSI_df = pd.DataFrame(columns=['5d_RSI', '10d_RSI', '14d_RSI', '30d_RSI', '90d_RSI', '180d_RSI'])
		RSI_df['5d_RSI'] = talib.RSI(self.df['Close'], timeperiod=5)
		RSI_df['10d_RSI'] = talib.RSI(self.df['Close'], timeperiod=10)
		RSI_df['14d_RSI'] = talib.RSI(self.df['Close'], timeperiod=14) # This is the standard
		RSI_df['30d_RSI'] = talib.RSI(self.df['Close'], timeperiod=30)
		RSI_df['90d_RSI'] = talib.RSI(self.df['Close'], timeperiod=90)
		RSI_df['180d_RSI'] = talib.RSI(self.df['Close'], timeperiod=180)
		return RSI_df

	def makeRecommendation(self, time_window):  
		rsi = talib.RSI(self.df['Close'])
		rsi = rsi.iloc[-1] # Get the most recent RSi value 
		if rsi < self.buyThreshold: 
			return 0 
		elif rsi > self.sellThreshold: 
			return 1
		else: 
			return 2