import pandas as pd
import os

#from QuantRecommendation import QuantRecommendation # Esta medio bobo como haces imports en Python no?


# Only thing needed is a dataframe of prices of a stock in the following format:
#
#| DATE | OPEN | CLOSE | VOLUME | HIGH | LOW |
#------------------------------------------------
#|OLDEST|      |       |        |      |     |
#|      |      |       |        |      |     |
#|      |      |       |        |      |     |
#|      |      |       |        |      |     |
#|      |      |       |        |      |     |
#|NEWEST|      |       |        |      |     |

# Order of the colummns does not matter, but it must contain at least those 6 columns, and the rows must be ordered: Oldest first row, newest row last
class QuantBackend:

    def __init__(self, df):

        self.df = df.copy(deep=False)

        recommendations = QuantRecommendation(self.df)

        self.QuantRecommendation, self.TurningRecommendation = recommendations.giveFinalRecommendation()

        self.rsi = recommendations.rsi
        self.ema = recommendations.ema
        self.macd = recommendations.macd
        self.obv = recommendations.obv

        self.rvol = recommendations.rvol
        self.ttm_squeeze = recommendations.ttm_squeeze

        #print("Quant recommendation is: {}, Turning recommendation is: {}".format(self.QuantRecommendation, self.TurningRecommendation))

# Remember, there are 2 Quant recommendations:

# The pure Quant recommendation which is used for 3 Type technical classifiers, i.e. those that can tell a Buy, hold or sell signal. Here are the encodings:
# 0 - Bullish
# 1 - Bearish
# 2 - Neutral
#
# The second recommendation, the Turning Recommednation as Javier mentioned are those technical classifiers that send 1 of 2 signals:
# A Hold signal which indicates now is not the time to enter the market and you should stay in your current position
# A move signal which indicates you should perform a trade IRREGARDELESS if your going long or short
#
# Here are the encodings:
#
# 0 - Hold
# 1 - Move