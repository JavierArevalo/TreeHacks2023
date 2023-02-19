import pandas as pd
import sys
import os

class ScrapeProcessor:

    #need 2 types of processors
    #one where stock name is only one word: below
    #one where stock name is two words: need to do
    #this is because reading by row names and the index change depending on the name since columns are added

    #List single name stocks: [Apple, Meta]
    #List compound name stocks: [Goldman Sachs]

    def __init__(self, stockName):
        self.stockName = stockName

    def processData2(self, stockName=None):
        if stockName is None:
            stockName = self.stockName

        if stockName is None:
            print("Error in processing scraper results. Could not find file for stock name: " + str(stockName))

        pathFile = "NewsData/" + str(stockName) + "News.csv"

        df = pd.read_csv(pathFile)
        return df
        #print(df)

    def processData(self, stockName=None):
        if stockName is None:
            stockName = self.stockName

        if stockName is None:
            print("Error in processing scraper results. Could not find file for stock name: " + str(stockName))
            return None

        pathFile = "NewsData/" + str(stockName) + "News.csv"

        df = pd.read_csv(pathFile)
        df.fillna(0, inplace=True)

        #print(df.to_markdown())
        #print(javi)

        #Create a dictionary
        #First key is stock Name
        #First value is a list of news object

        #Dont use title 1 for name, rather use stock name since that might only be first word of complete stock name

        newsList = []
        index = 0
        #df = [row ]
        for row in df.iterrows():
            #print(row)
            #print(len(row))
            #print(type(row))
            #print(index)
            if (index == 0):
                index = index + 1
                continue
            #print("Here")
            index = index + 1
            newsObject = {}
            row = row[1]
            newsObject["StockName"] = str(stockName)
            #values = [str(stockName), row[3], row[5], row[2]]
            #Need to change below ones to integer values of column name
            #print("")
            #print("")
            #print("Row values")
            #print("Row value 0")
            #print(row[0])
            #print("")
            #print("Row value 1")
            #print(row[1])
            #print("")
            #print(row[2])
            newsObject["NewsHeader"] = row["Title"]
            newsObject["News"] = row["View"]
            #print(newsObject)
            #print(values)
            newsObject["Url"] = row["Title_URL"]
            newsList.append(newsObject)
            #print(newsObject)

        finalDictForStock = {}
        finalDictForStock[str(stockName)] = newsList
        #print(finalDictForStock)
        return finalDictForStock
