import pandas as pd 
import numpy as np 
import talib # Technical Indicator Library

TIME_WINDOW = 14 # In days
MINIMUM_THRESHOLD = 0.5 # The minimum ratio between volume/OBV for the signal to be considered significant

class OBV:

	def __init__(self, df):
		self.df = df.copy(deep=False)

	def findOBV(self, time_window):
		#why negative time window?
		obv = talib.OBV(self.df['Close'].iloc[-time_window:], self.df['Volume'].iloc[-time_window:])
		ratio = self.df['Volume'].iloc[-time_window:]/obv
		difference = obv - self.df['Volume'].iloc[-time_window]
		difference = np.sign(difference).astype(int)
		obv_df = pd.concat([difference, ratio], axis=1) # Put difference and ratio into one df 
		obv_df.columns = ['Difference', 'Ratio']
		return obv_df

	def makeRecommendation(self):
		difference = self.findOBV(TIME_WINDOW)["Difference"].iloc[-1] #Get the last entry of specified time period 
		ratio = self.findOBV(TIME_WINDOW)['Ratio'].iloc[-1]
		if ratio < MINIMUM_THRESHOLD and difference == 1: # Bullish Signal 
			return 0
		elif ratio < MINIMUM_THRESHOLD and difference == -1: # Bearish Signal 
			return 1 
		else:
			return 2 # Neutral Signal