
import pandas as pd 

ART_MULTIPLIER = 1.5
PERIOD = 20
STD_MULTIPLIER = 2

class TTM_SQUEEZE:
	def __init__(self, df):
		self.df = df.copy(deep=False)

		self.df['sma'] = df['Close'].rolling(window=PERIOD).mean()
		self.df['stddev'] = df['Close'].rolling(window=PERIOD).std()

		self.df['lower_band'] = self.df['sma'] - (STD_MULTIPLIER * self.df['stddev'])
		self.df['upper_band'] = self.df['sma'] + (STD_MULTIPLIER * self.df['stddev'])

		self.df['TR'] = abs(self.df['High'] - self.df['Low'])
		self.df['ATR'] = self.df['TR'].rolling(window=PERIOD).mean()

		self.df['lower_keltner'] = self.df['sma'] - (self.df['ATR'] * ART_MULTIPLIER)
		self.df['upper_keltner'] = self.df['sma'] + (self.df['ATR'] * ART_MULTIPLIER)

	def makeRecommendation(self):
		self.df['squeeze_on'] = self.df.apply(in_squeeze, axis=1)
		if self.df.iloc[-3]['squeeze_on'] and not self.df.iloc[-1]['squeeze_on']:
			# Return 1, a move signal
			return 1
		else:
			return 0

	def in_squeeze(df):
		return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']