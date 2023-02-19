import pandas as pd 
import numpy as np 

SHORT_TERM = 30
LONG_TERM = 60


class EMA:

	def __init__(self, df):
		self.df = df.copy(deep=False)
		#Simple Moving Average
		self.SMA_df = self.calculateSMA()
		#Exponential Moving Average
		self.EMA_df = self.calculateEMA()
		self.strategy_df = self.calculateStrategy(SHORT_TERM, LONG_TERM) # use default short and long term 

	def calculateSMA(self):
		SMA_df = pd.DataFrame(columns=['30d_SMA', '50d_SMA', '60d_SMA', '150d_SMA', '253d_SMA'])
		SMA_df['30d_SMA'] = np.round(self.df['Close'].rolling(30).mean(), 2)
		SMA_df['50d_SMA'] = np.round(self.df['Close'].rolling(50).mean(), 2)
		SMA_df['60d_SMA'] = np.round(self.df['Close'].rolling(60).mean(), 2)
		SMA_df['150d_SMA'] = np.round(self.df['Close'].rolling(150).mean(), 2)
		SMA_df['253d_SMA'] = np.round(self.df['Close'].rolling(253).mean(), 2)
		return SMA_df

	def calculateEMA(self):
		EMA_df = pd.DataFrame(columns=['30d_EMA', '50d_EMA', '60d_EMA', '150d_EMA', '253d_EMA'])
		EMA_df['30d_EMA'] = np.round(self.df['Close'].ewm(30).mean(), 2)
		EMA_df['50d_EMA'] = np.round(self.df['Close'].ewm(50).mean(), 2)
		EMA_df['60d_EMA'] = np.round(self.df['Close'].ewm(60).mean(), 2)
		EMA_df['150d_EMA'] = np.round(self.df['Close'].ewm(150).mean(), 2)
		EMA_df['253d_EMA'] = np.round(self.df['Close'].ewm(253).mean(), 2)

	def EMAvsSMA(self, time_window):
		vs_df = pd.DataFrame()
		vs_df['EMA'] = np.round(self.df['Close'].ewm(time_window).mean(), 2)
		vs_df['SMA'] = np.round(self.df['Close'].rolling(time_window).mean(), 2)
		vs_df['Strategy'] = np.where( (vs_df['EMA'] > vs_df['SMA']), "EMA", "SMA")
		return vs_df

	def calculateStrategy(self, short_term, long_term):
		results_df = pd.DataFrame(columns=['Short_Term','Long_Term','Strategy'])
		results_df['Short_Term'] =  np.round(self.df['Close'].ewm(short_term).mean(), 2)
		results_df['Long_Term'] =  np.round(self.df['Close'].ewm(long_term).mean(), 2)
		results_df['Strategy'] = np.where((results_df['Short_Term'] > results_df['Long_Term']), 0, 1) # Decide Buy or Sell
		results_df.loc[(results_df['Short_Term'] == results_df['Long_Term']), ["Strategy"]] = 2 # Decide Neutral
		return results_df

	def makeRecommendation(self):
		return self.strategy_df["Strategy"].iloc[-1] #Get most recent prediction 



