import pandas as pd
import talib

from datetime import date 
from datetime import datetime 
from dateutil.relativedelta import relativedelta

# Type 3 Technical Indicators
# What does type 3 mean?
from RSI import RSI
from EMA import EMA
from MACD import MACD
from OBV import OBV

#Type 2 Technical Indicators
#What does Type 2 Indicators mean?
from RVOL import RVOL
from TTM_SQUEEZE import TTM_SQUEEZE

#New added models: Fama French models
from FamaFrench3FactorModel import FamaFrench3FactorModel
from FamaFrench5FactorModel import FamaFrench5FactorModel


from FinancialDataAPI import FinancialDataAPI

import numpy as np
import sys
import os
import json
from FinancialDataAPI import FinancialDataAPI as financialAPI



#For saving ML models
import pickle

import json

from jsonmerge import merge

#from SentimentAnalysisBackend import SentimentAnalysisBackend
#from QuantBackend import QuantBackend
from ScrapeProcessor import ScrapeProcessor

#import firebase_admin
#from firebase_admin import credentials, firestore
#from firebase_admin import db
#from google.cloud import storage

#from firebase import firebase

#storage_client = storage.Client()

#cred_path = os.path.join(os.getcwd(), "minervaJavi.json")
#cred = credentials.Certificate(cred_path)
#cred = credentials.ApplicationDefault()
#app = firebase_admin.initialize_app(cred, {'databaseURL': 'https://minerva-a566a-default-rtdb.firebaseio.com/'})
#print("Created firebase credentials for Minerva First")
#ef = db.reference('/')

#firebase_app = firebase.FirebaseApplication('https://minerva-a566a-default-rtdb.firebaseio.com/', None)

end = date.today()
start = end - relativedelta(years=1)

class QuantBackendComplete:

    def __init__(self, stockName, stockTicker):
        self.stockName = stockName
        self.stockTicker = stockTicker
        self._financialDataAPI = FinancialDataAPI(stockTicker)
        done = self.getPriceData()
        #self.updatePrices()

    #Might be repeating every 25 times, might be a better way to optimize it by stock
    def updatePricesNew():
        return 0

    def updatePrices(self, stockTicker):
        #New changes
        #_financialAPI = financialAPI(stockTicker)
        #stock_df = _financialAPI.getPrices(stockTicker)

        stock_df = self.dataFrames

        #self.dataFrames

        index = 0
        listPrices = []
        megaDict = {}
        for idx, row in stock_df.iterrows():

            dictCur = {}
            date = row['Date']
            price = row['Close']
            dictCur[str(date)] = price

            listPrices.append(dictCur)

            megaDict[str(date)] = price
            index = index + 1

        listPrices = json.dumps(listPrices)

        dictFinal = {}
        dictFinal[str(stockTicker)] = megaDict

        self.putToDatabasePrices(self, dictFinal, stockTicker)

    def putToDatabasePrices(self, dectFinal, stockTicker):
        #Make sure it is updating for every stock name
        #For every day, every supported stock should be updated
        key = ticker
        postAttempt = firebase_app.put('/stocksPrices', str(key), json)



    def getPriceData(self):
        _dataDF = self._financialDataAPI.getPrices(self.stockTicker)
        #_dataJSON = self._financialDataAPI.processPrices(_dataDF)

        self.dataFramePrices = _dataDF
        return 1
        #self.jsonPrices = _dataJSON

    def getRSI(self, window):
        #double check if this are the ideal thresholds
        #can check thresholds with machine learning later on
        recommendation = RSI(30, 70, self.dataFramePrices).makeRecommendation(window)
        #_rsiClass.createRSI()
        #recommendation = _rsiClass.makeRecommendation(14)
        return recommendation

    def getEMA(self):
        recommendation = EMA(self.dataFramePrices).makeRecommendation()
        return recommendation

    def getMACD(self):
        recommendation = MACD(self.dataFramePrices).makeRecommendation()
        return recommendation

    def getOBV(self):
        recommendation = OBV(self.dataFramePrices).makeRecommendation()
        return recommendation

    def getRVOL(self, window):
        recommendation = RVOL(2, self.dataFramePrices).makeRecommendation(window)
        return recommendation

    def getTTMSQUEEZE(self):
        recommendation = TTM_SQUEEZE(self.dataFramePrices).makeRecommendation()
        return recommendation

    #New models 
    def getFamaFrench3Factor(self):
        ff3f = FamaFrench3FactorModel()
        monthly_return, yearly_return = ff3f.getPrediction(self.stockTicker, start, end)
        return monthly_return, yearly_return

    def getFamaFrench5Factor(self):
        ff5f = FamaFrench5FactorModel()
        monthly_return, yearly_return = ff5f.getPrediction(self.stockTicker, start, end)
        return monthly_return, yearl

    def getAllFinancialIndicators(self):

        #list in order of [RSI, EMA, MACD, OBV, RVOL, and TTM_SQUEEZE]
        #Note first 4 indicators are type 2 and last 2 indicators are type 3

        #Type 2 Financial Indicators
        rsi = self.getRSI(14)
        ema = self.getEMA()
        macd = self.getMACD()
        obv = self.getOBV()

        #Type 3 Financial Indicators
        rvol = self.getRVOL(14)
        ttm_squeeze = self.getTTMSQUEEZE()

        #New Fama French Models
        ff3Monthly, ff3Yearly = self.getFamaFrench3Factor()
        ff5Monthly, ff5Yearly = self.getFamaFrench5Factor()


        indicators = [rsi, ema, macd, obv, rvol, ttm_squeeze, ff3Monthly, ff3Yearly, ff5Monthly, ff5Yearly]
        return indicators

    def getFinalClassification(self, indicators):
        most_frequent = self.most_frequent(indicators[0:4])
        return most_frequent


    def most_frequent(self, list):
        return max(set(list), key = list.count)