

class RVOL:

	def __init__(self, thresholdParam, df):
		self.rvolThreshold = thresholdParam
		self.df = df.copy(deep=False)

	def calculateAverageVolume(self):
		df = pd.DataFrame(columns=['5d_AV','10d_AV','30d_AV','90d_AV','180d_AV'])
		df['5d_AV'] = np.round(self.df['Volume'].rolling(5).mean(),2)
		df['10d_AV'] = np.round(self.df['Volume'].rolling(10).mean(),2)
		df['30d_AV'] = np.round(self.df['Volume'].rolling(30).mean(),2)
		df['90d_AV'] = np.round(self.df['Volume'].rolling(90).mean(),2)
		df['180d_AV'] = np.round(self.df['Volume'].rolling(180).mean(),2)
		return df

	def calculateSTD(self):
		df = pd.DataFrame(columns=['5d_SDV','10d_SDV','30d_SDV','90d_SDV','180d_SDV'])
		df['5d_SDV'] = np.round(self.df['Volume'].rolling(5).std(),2)
		df['10d_SDV'] = np.round(self.df['Volume'].rolling(10).std(),2)
		df['30d_SDV'] = np.round(self.df['Volume'].rolling(30).std(),2)
		df['90d_SDV'] = np.round(self.df['Volume'].rolling(90).std(),2)
		df['180d_SDV'] = np.round(self.df['Volume'].rolling(180).std(),2)
		return df 

	def calculateRVOL(self):
		df = pd.DataFrame(columns=['5d_RVOL','10d_RVOL','30d_RVOL','90d_RVOL','180d_RVOL'])
		std_df = calculateSTD()
		df['5d_RVOL'] = np.round(self.df['Volume']/std_df['5d_SDV'], 2)
		df['10d_RVOL'] = np.round(self.df['Volume']/std_df['10d_SDV'], 2)
		df['30d_RVOL'] = np.round(self.df['Volume']/std_df['30d_SDV'], 2)
		df['90d_RVOL'] = np.round(self.df['Volume']/std_df['90d_SDV'], 2)
		df['180d_RVOL'] = np.round(self.df['Volume']/std_df['180d_SDV'], 2)
		return df

	def makeRecommendation(self, time_window): # time window in days 
		std = np.round(self.df['Volume'].rolling(time_window).std(), 2)
		rvol = np.round(self.df['Volume'] / std, 2)
		result = rvol.iloc[-1] # Get the msot recent RVOL value 
		if result >= self.rvolThreshold:
			return 1 # A moe signal 
		else: 
			return 0
