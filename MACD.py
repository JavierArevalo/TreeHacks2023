import pandas as pd 
import talib #Technical Indicator Library


# The MACD's popularity is largely due to its ability to help quickly spot increasing short-term momentum.
# Traders will commonly rely on the default settings of 12- and 26-day periods.
# signal line = macdsignal
# macd - signal line = macdhist
FAST_PERIOD = 12 # Default for traders
SLOW_PERIOD = 26 # Default for traders
SIGNAL_PERIOD = 9 # Default for traders
NEUTRAL_THRESHOLD = 1 # The difference between signal line and MACD has to be at least this big to provoke a buy or sell signal

class MACD:

	def __init__(self, df):
		self.df = df.copy(deep=False)
		macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=FAST_PERIOD, slowperiod=SLOW_PERIOD, signalperiod=SIGNAL_PERIOD)
		self.macd = macd 
		self.macdsignal = macdsignal
		self.macdhist = macdhist

	def makeRecommendation(self):
		x = self.macdhist.iloc[-1] #Most recent macdhisst 
		if x < -NEUTRAL_THRESHOLD:
			return 1 # Bearish
		elif x > NEUTRAL_THRESHOLD:
			return 0 # Bullish 
		else: 
			return 2 #Netural
