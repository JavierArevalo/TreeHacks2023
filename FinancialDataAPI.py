#Code to retrieve price data for stocks using my financial API
#Returns object in form of {'ticker': [{'date1': price1}, ..., {'daten', pricen}
#Now returns dataframe of prices

import requests
import json
import pandas as pd

class FinancialDataAPI:
    def __init__(self, ticker):
        self.ticker = ticker
        self.access_key = 'ef7d7d5ef00d2cc85d76913392c86b94'
        self.baseRequestURL = 'https://api.marketstack.com/v1/tickers'
        #don't forget to add url part of /eod after /'ticker'

    def getPrices(self, ticker):
        self.ticker = (str(ticker)).lower()
        params = {
            "access_key": self.access_key,
        }
        completeURL = self.baseRequestURL + '/' + str(self.ticker) + '/eod'
        api_result = requests.get(completeURL, params)
        api_response = api_result.json()
        print("API Response")
        print(api_response)
        print("")
        processedPrices = self.processPrices(api_response)

        #Turn from nested json into list of single json entries and then list to pandas dataframe
        df = pd.DataFrame.from_dict(processedPrices)
        print ("Data Frame: ")
        print(df)
        return df

    def processPrices(self, priceData):
        print("Price Data")
        print(priceData)
        data = priceData['data']
        name = data['name']
        ticker = data['symbol']
        prices = data['eod']
        #print(name)
        #print(ticker)
        #print("Price Data: ")
        #print(prices)
        #stock_df = pd.read_json(priceData, orient='index')
        listPrices = []
        for idx, row in enumerate(prices):
            dictCur = {}
            date = row['date']
            price = row['close']
            high = row['high']
            low = row['low']
            volume = row['volume']
            dictCur['Date'] = str(date)
            dictCur['Close'] = price
            dictCur['Volume'] = volume
            dictCur['High'] = high
            dictCur['Low'] = low
            listPrices.append(dictCur)
        return listPrices




if __name__ == "__main__":
    sampleTicker = 'GS'
    _financialDataAPI = FinancialDataAPI(sampleTicker)
    priceData = _financialDataAPI.getPrices(sampleTicker)
    #processedPrices = _financialDataAPI.processPrices(priceData)