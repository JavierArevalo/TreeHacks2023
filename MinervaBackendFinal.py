
import numpy as np
import pandas as pd
import sys
import os
import json
from FinancialDataAPI import FinancialDataAPI as financialAPI

#from QuantBackendComplete import QuantBackendComplete

from SentimentModelBERT import SentimentModelBERT


#For saving ML models
import pickle

import json

from jsonmerge import merge

#from SentimentAnalysisBackend import SentimentAnalysisBackend
from QuantBackend import QuantBackend
from ScrapeProcessor import ScrapeProcessor

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import db
from google.cloud import storage

#from firebase import firebase

#storage_client = storage.Client()/Users/javierarevalo/Desktop/Minerva/MinervaMVP/API_Authentication/minervaJavi.json

cred_path = os.path.join(os.getcwd(), "minervaJavi.json")
cred = credentials.Certificate(cred_path)
cred = credentials.ApplicationDefault()
app = firebase_admin.initialize_app(cred, {'databaseURL': 'https://minerva-a566a-default-rtdb.firebaseio.com/'})
#print("Created firebase credentials for Minerva First")
ef = db.reference('/')

firebase_app = firebase.FirebaseApplication('https://minerva-a566a-default-rtdb.firebaseio.com/', None)

STOCK_DF_PATH = "/Technical_Indicator/datasets/AAPL.csv" # This is relative to MinervaMVP Directory

categories = {
    "Amazon": "Tech",
    "Apple": "Tech",
    "BankofAmerica": "Banking",
    "BeyondPetroleum": "Oil",
    "CarnivalCorporations": "Travel",
    "Delta": "Travel",
    "Meta": "Tech",
    "GeneralMotors": "Automotive",
    "GoldmanSachs": "Banking",
    "Google": "Tech",
    "Hilton": "Hotels",
    "Honda": "Automotive",
    "JohnsonandJohnson": "Pharmaceutical",
    "JPMorgan": "Banking",
    "Microsoft": "Tech",
    "Netflix": "Tech",
    "NVIDIA": "Tech",
    "Pfizer": "Pharmaceutical",
    "Tesla": "Tech",
    "Twitter": "Tech",
    "UBS": "Banking",
    "United": "Travel",
    "VertexPharmaceutical": "Pharmaceutical",
    "AdvancedMicroDevices": "Tech",
}


def postToDatabase(jsonRes):
    result = firebase_app.post('/stocksMinervaMVP', jsonRes)
    return result


def postToDatabaseFav(jsonFavoriteUser):
    user = None
    for key in jsonFavoriteUser:
        user = key
    result = firebase_app.put('/userFavorites', str(user), jsonFavoriteUser)


def putToDatabase(jsonRes):
    key = str(jsonRes["stock_name"]) + "Final"
    postAttempt = firebase_app.put('/stocksMinervaMVP', str(key), jsonRes)


def putToDatabasePrices(json, ticker):
    key = ticker
    postAttempt = firebase_app.put('/stocksPrices', str(key), json)


def processPrices(stockTicker):

    #New changes
    _financialAPI = financialAPI(stockTicker)
    stock_df = _financialAPI.getPrices(stockTicker)

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

    putToDatabasePrices(dictFinal, stockTicker)


def processNews(stockTicker):

    path = os.getcwd() + '/DataBackend/News/' + str(stockTicker) + '_results.csv'
    news_df = pd.read_csv(path, delimiter=',')
    megaDict = {}
    for idx, row in news_df.iterrows():
        curTime = row['Time']
        curHeadline = row['Headline']
        curSource = row['Source']
        curURL = row['URL']
        curDict = {}
        curDict['time'] = curTime
        curDict['headline'] = curHeadline
        curDict['source'] = curSource
        curDict['url'] = curURL
        nameTag = 'News' + str(idx)
        megaDict[nameTag] = curDict
    dictFinal = {}
    dictFinal[str(stockTicker)] = megaDict
    postToDatabaseNews(dictFinal)


def postToDatabaseNews(news):
    result = firebase_app.post('/News', news)


def postToDatabasePrices(prices, key=None):
    for price in prices:
        key=price
    result = firebase_app.put('/stocksPrices', str(key), prices)


def set_default_firebase(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def databaseTryHardCoded():
    minerva_backend2 = {
        "stock_name": "MinervaLatestOctober2021",

        "sentiment_classification": "Bearish",

        "bullish_p": 2200,
        "bearish_p": 22000,
        "neutral_p": 5200,

        "bayesian_classification": "Bullish",
        "svm_classification": "Bearish",

        "quant_classification": "Bearish",

        "rvol": "sell",
        "rsi": "sell",
        "ttm": "sell",
        "ema": "sell",

        "id": "mockDatabaseSetup1.2",

        "final_score": "bearish"
    }

    result = postToDatabase(minerva_backend2)


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
      if isinstance(obj, set):
         return list(obj)
      if isinstance(obj, np.integer):
         return int(obj)
      if isinstance(obj, np.floating):
         return float(obj)
      if isinstance(obj, np.ndarray):
         return obj.tolist()
      return json.JSONEncoder.default(self, obj)

#Input should be classifications from each of the classifiers {1, 3, 4, 5, 6}
def predict_fundamental_NN_Sentiment(trainedNN, classification1, classification2, classification3, classification4, classification5, classification6):

    listInputPredict = [int(classification1), int(classification2), int(classification3), int(classification4), int(classification5), int(classification6)]

    #Remove the following once Vector Distance Classifier works
    listInputPredict = [int(classification1), int(classification3), int(classification4), int(classification5), int(classification6)]

    res = trainedNN.predict([listInputPredict])
    return res

class MinervaBackend:

    def __init__(self, tweet):
        self.tweet = tweet

    # This method returns the trained NN from the SentimentAnalysisBackend
    # This will be used to get our official predictions of sentiment of our supported stocks based on the most recenet 30 day 100 tweets of all of our supported stocks
    def getTrainedSentimentNN(self):
        path_df_test = "/Users/mau/MinervaMVP/Data/Stock_Tweet_Data/mainDataset.csv"

        path_df_test = os.path.join(os.getcwd() + '/DataPrelabeled/mainDataset.csv')
        df_test =  pd.read_csv(path_df_test)

        _sentimentAnalysisBackend = SentimentAnalysisBackend(tweet) # Do the classification for 1 tweet
        #_sentimentAnalysisBackend.start()

        #trained Neural netork ready to make predictions:

        #trainedNN, prelabeledTweets, naiveClassifier, vectorDistanceClassifier, discriminantBasedClassifier, adjectiveAdverbPhraseClassifier = _sentimentAnalysisBackend.getTrainedNeuralNetwork()
        return []
        #get trained Neural Network ready to make predictions
        return _sentimentAnalysisBackend, trainedNN, prelabeledTweets, naiveClassifier, vectorDistanceClassifier, discriminantBasedClassifier, adjectiveAdverbPhraseClassifier

    #This method classifies the tweets for a given month for ONE company
    #Return metrics needed to upload data from SentimentAnalysis Backend
    #MISSING: code to get bayesian and svm classification
    def classifyTweetsLastMonth2(self, companyFileNameJSON, _sentimentAnalysisBackend, trainedNN, prelabeledTweets, naiveClassifier, vectorDistanceClassifier, discriminantBasedClassifier, adjectiveAdverbPhraseClassifier):
        #TODO: open JSON file
        #print("FIRST TIME OPENING A JSON")
        with open (companyFileNameJSON) as file:
            processedFile = file.read()
            #passing in a string
            data = json.loads(processedFile)

        countBullish = int(0)
        countBearish = int(0)
        countNeutral = int(0)

        listTweets = data['data']
        for tweetJSONEntry in listTweets:
            tweet = tweetJSONEntry["text"]

            #CALL invididual classification
            classification = _sentimentAnalysisBackend.individualClassification(tweet, prelabeledTweets, naiveClassifier, vectorDistanceClassifier, discriminantBasedClassifier, adjectiveAdverbPhraseClassifier)
            #returns array of form ["NaiveClassifierClassification", "DiscriminantBasedClassification", "AdjectiveAdverbPhraseClassification"]

            predictionsData = []
            for predictionClassifier in classification:
                predictionsData.append(predictionClassifier)

            #TODO
            #find a way to get bayesian and svm classification from _sentimentAnalysisBackend ideally through a method call
            #classificationBayesian = [MISSING]
            classificationBayesian = 2
            #classificationSVM = [MISSING]
            classificationSVM = 2

            predictionsData.append(classificationBayesian)
            predictionsData.append(classificationSVM)

            #Once I have complete predictionsData, can call predict on trained NN
            finalClassificationTweet = predict_fundamental_NN_Sentiment(trainedNN, predictionsData[0], 0, predictionsData[1], predictionsData[2], predictionsData[3], predictionsData[4])
            #Now see if it is bearish bullish or neutral
            if (int(finalClassificationTweet) == 1):
                countBullish = countBullish + 1
            elif (int(finalClassificationTweet) == 2):
                countBearish = countBearish + 1
            else:
                countNeutral = countNeutral + 1

        #Dont include Bayesian classification and svm classifiction in database
        totalTweets = countBullish + countBearish + countNeutral
        bullishP = float(countBullish/totalTweets)
        bearishP = float(countBearish/totalTweets)
        neutralP = float(countNeutral/totalTweets)

        #finalClassification of stock is argmax(bullishP, bearishP, neutralP)
        finalClassificationStock = "Neutral"
        if (bullishP >= bearishP and bullishP >= neutralP):
            finalClassificationStock = "Bullish"
        elif (bearishP >= bullishP and bearishP >= neutralP):
            finalClassificationStock = "Bearish"
        #Else it means that neutralP is the largest percentage so leave finalClassificationStock as "Neutral"

        return bullishP, bearishP, neutralP, finalClassificationStock

        #Metrics needed:
        #bullishP = countBullish / (countBullish+countBearish+countNeutral)
        #barishP = countBearish / (countBullish+countBearish+countNeutral)
        #neutralP = countNeutral / (countBullish+countBearish+countNeutral)

        #Sentiment classification: argmax(countBullish, countBearish, countNeutral)

    def classifyTweetLastMonth(self, companyFileNameJSON, _sentimentAnalysisBackend):
        #print("FIRST TIME OPENING A JSON")
        with open (companyFileNameJSON) as file:
            processedFile = file.read()
            #passing in a string
            data = json.loads(processedFile)

        countBullish = int(0)
        countBearish = int(0)
        countNeutral = int(0)

        listTweets = data['data']
        counter = int(0)
        for tweetJSONEntry in listTweets:
            if (type(tweetJSONEntry) is list):
                break
            else:
                tweet = tweetJSONEntry["text"]

                #TODO: pass SentimentAnalysisInstance as parameter to not initialize all models for each tweet I am analyzing
                # Do the classification for 1 tweet
                ##print("SENTIMENT PREDICTION IS: ")
                prediction = _sentimentAnalysisBackend.makeRecommendation(tweet)
                if (int(prediction[0]) == 1):
                    countBullish = countBullish + 1
                elif (int(prediction[0]) == 2):
                    countBearish = countBearish + 1
                else:
                    countNeutral = countNeutral + 1
            counter = counter + 1


        totalTweets = countBullish + countBearish + countNeutral
        bullishP = float(countBullish/totalTweets)
        bearishP = float(countBearish/totalTweets)
        neutralP = float(countNeutral/totalTweets)

        finalClassificationStock = 0
        if (bullishP >= bearishP and bullishP >= neutralP):
            finalClassificationStock = 1
        elif (bearishP >= bullishP and bearishP >= neutralP):
            finalClassificationStock = 2
        return bullishP, bearishP, neutralP, finalClassificationStock

    def classifyNews(self, newsObject, stockName, _sentimentAnalysisBackend):

        countBullish = int(0)
        countBearish = int(0)
        countNeutral = int(0)

        counter = int(0)

        for entry in newsObject[str(stockName)]:
            news = entry["News"]

            #TODO: double check that below code works same with news vs with tweet
            prediction = _sentimentAnalysisBackend.makeRecommendation(news)

            posScore = prediction['positive']
            negScore = prediction['negative']
            neutralScore = prediction['neutral']

            if posScore >= negScore and posScore >= neutralScore:
                countBullish = countBullish + 1 
            elif negScore > posScore and negScore > neutralScore:
                countBearish = countBearish + 1 
            else: 
                countNeutral = countNeutral + 1

            #possible change: pass header instead of news itself, but explore this alternative later on

            #if (int(prediction[0]) == 1):
                #countBullish = countBullish + 1
            #elif (int(prediction[0]) == 2):
                #countBearish = countBearish + 1
            #else:
                #countNeutral = countNeutral + 1
            counter = counter + 1

        totalNews = countBullish + countBearish + countNeutral
        bullishP = float(countBullish/totalNews)
        bearishP = float(countBearish/totalNews)
        neutralP = float(countNeutral/totalNews)

        finalClassificationStock = 0
        if (bullishP >= bearishP and bullishP >= neutralP):
            finalClassificationStock = 1
        elif (bearishP >= bullishP and bearishP >= neutralP):
            finalClassificationStock = 2

        return bullishP, bearishP, neutralP, finalClassificationStock

    def getSentimentBackendBERT(self):
        listFilenames = ['FacebookLastMonth.json', 'AppleLastMonth.json', 'NetflixLastMonth.json', 'GoogleLastMonth.json', 'MicrosoftLastMonth.json', 'AmazonLastMonth.json', 'TeslaLastMonth.json', 'GoldmanSachsLastMonth.json', 'JPMorganLastMonth.json', 'UBSLastMonth.json', 'BankOfAmericaLastMonth.json', 'PfizerLastMonth.json', 'JohnsonAndJohnsonLastMonth.json', 'GeneralMotorsLastMonth.json', 'HiltonLastMonth.json', 'DeltaLastMonth.json', 'UnitedLastMonth.json', 'NVIDIALastMonth.json', 'TwitterLastMonth.json', 'VertexPharmaceuticalLastMonth.json', 'ADMLastMonth.json', 'HondaLastMonth.json', 'CarnivalCorporationsLastMonth.json', 'BeyondPetroleumLastMonth.json']
        listStockNames = ['Meta', 'Apple', 'Netflix', 'Google', 'Microsoft', 'Amazon', 'Tesla', 'GoldmanSachs', 'JPMorgan', 'UBS', 'BankofAmerica', 'Pfizer', 'JohnsonandJohnson', 'GeneralMotors', 'Hilton', 'Delta', 'United', 'NVIDIA', 'Twitter', 'VertexPharmaceutical', 'AdvancedMicroDevices', 'Honda', 'CarnivalCorporations', 'BP']

        listStockTickers = ['META', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'GS', 'JPM', 'UBS', 'BAC', 'PFE', 'JNJ', 'GM', 'H', 'DAL', 'UAL', 'NVDA', 'TWTR', 'VRTX', 'AMD', 'HMC', 'CCL', 'BP']

        _sentimentAnalysisBackend = SentimentModelBert()

        listJSONS = []

        index = int(0)

        for stockName in listStockNames:

            _scrapeProcessor = ScrapeProcessor(stockName)

            processedData = _scrapeProcessor.processedData()

            bullishP, bearishP, neutralP, sentiment_classification = self.classifyNews(processedData, stockName, _sentimentAnalysisBackend)

            stockTicker = listStockTickers[index]

            # double check classify News to make sure I return the correct ones 
            stockBackend = {}
            stockBackend = {
                "stock_name": str(stockName),
                "stock_ticker": str(stockTicker),
                "sentiment_classification": str(sentiment_classification),
                "bullish_p": str(bullishP),
                "bearish_p": str(bearishP),
                "neutral_p": str(neutralP)
            }
            listJSONS.append(stockBackend)

            index = index + 1
        return listJSONS






    def getSentimentBackend(self):
        listFilenames = ['FacebookLastMonth.json', 'AppleLastMonth.json', 'NetflixLastMonth.json', 'GoogleLastMonth.json', 'MicrosoftLastMonth.json', 'AmazonLastMonth.json', 'TeslaLastMonth.json', 'GoldmanSachsLastMonth.json', 'JPMorganLastMonth.json', 'UBSLastMonth.json', 'BankOfAmericaLastMonth.json', 'PfizerLastMonth.json', 'JohnsonAndJohnsonLastMonth.json', 'GeneralMotorsLastMonth.json', 'HiltonLastMonth.json', 'DeltaLastMonth.json', 'UnitedLastMonth.json', 'NVIDIALastMonth.json', 'TwitterLastMonth.json', 'VertexPharmaceuticalLastMonth.json', 'ADMLastMonth.json', 'HondaLastMonth.json', 'CarnivalCorporationsLastMonth.json', 'BeyondPetroleumLastMonth.json']
        listStockNames = ['Meta', 'Apple', 'Netflix', 'Google', 'Microsoft', 'Amazon', 'Tesla', 'GoldmanSachs', 'JPMorgan', 'UBS', 'BankofAmerica', 'Pfizer', 'JohnsonandJohnson', 'GeneralMotors', 'Hilton', 'Delta', 'United', 'NVIDIA', 'Twitter', 'VertexPharmaceutical', 'AdvancedMicroDevices', 'Honda', 'CarnivalCorporations', 'BP']

        listStockTickers = ['META', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'GS', 'JPM', 'UBS', 'BAC', 'PFE', 'JNJ', 'GM', 'H', 'DAL', 'UAL', 'NVDA', 'TWTR', 'VRTX', 'AMD', 'HMC', 'CCL', 'BP']


        _sentimentAnalysisBackend = SentimentAnalysisBackend()

        listJSONS = []

        #For now run only with apple
        #listFilenames = ['AppleLastMonth.json', 'MicrosoftLastMonth.json', 'TeslaLastMonth.json', 'GoldmanSachsLastMonth.json']
        #listStockNames = ['Apple', 'Microsoft', 'Tesla', 'Goldman Sachs', 'Facebook']
        index = int(0)

        #for filename in listFilenames:
        for stockName in listStockNames:

            #addition
            _scrapeProcessor = ScrapeProcessor(stockName)

            #Modifications:
            #Initially passing a filename of a json file. e.g. FacebookLastMonth.json
            #Code before (not broken version):
            #bullishP, bearishP, neutralP, sentiment_classification = self.classifyTweetLastMonth(filename, _sentimentAnalysisBackend)

            #NEW:


            #retrieve json object:
            processedData = _scrapeProcessor.processData()

            #Now will pass a big json object that contains a list where each entry is a news (instead of a tweet)
            #CHANGE: file json object directly instead of filename. Also need to pass name of the stock
            bullishP, bearishP, neutralP, sentiment_classification = self.classifyNews(processedData, stockName, _sentimentAnalysisBackend)

            stockTicker = listStockTickers[index]

            stockBackend = {}
            stockBackend = {
                "stock_name" : str(stockName),
                "stock_ticker": str(stockTicker),
                "sentiment_classification": str(sentiment_classification),
                "bullish_p": str(bullishP),
                "bearish_p": str(bearishP),
                "neutral_p": str(neutralP),
            }
            ##print("Stock backend json: ")
            ##print(stockBackend)
            listJSONS.append(stockBackend)

            index = index + 1
        return listJSONS



    # For now make it return a list of jsons to post to firebase, one for each company
    # Each json file in list must be edited and added the data from the quant backend before posting it to firebase
    def getSentimentBackend2(self):
        listFilenames = ['FacebookLastMonth.json', 'AppleLastMonth.json', 'NetflixLastMonth.json', 'GoogleLastMonth.json', 'MicrosoftLastMonth.json', 'AmazonLastMonth.json', 'TeslaLastMonth.json', 'GoldmanSachsLastMonth.json', 'JPMorganLastMonth.json', 'UBSLastMonth.json', 'BankOfAmericaLastMonth.json', 'PfizerLastMonth.json', 'JohnsonAndJohnsonLastMonth.json', 'UberLastMonth.json', 'GeneralMotorsLastMonth.json', 'HiltonLastMonth.json', 'DeltaLastMonth.json', 'UnitedLastMonth.json', 'NVIDIALastMonth.json', 'TwitterLastMonth.json', 'VertexPharmaceuticalLastMonth.json', 'ADMLastMonth.json', 'HondaLastMonth.json', 'CarnivalCorporationsLastMonth.json', 'BeyondPetroleumLastMonth.json']
        listStockNames = ['Facebook', 'Apple', 'Netflix', 'Google', 'Microsoft', 'Amazon', 'Tesla', 'Goldman Sachs', 'JPMorgan', 'UBS', 'Bank of America', 'Pfizer', 'Johnson and Johnson', 'Uber', 'General Motors', 'Hilton', 'Delta', 'United', 'NVIDIA', 'Twitter', 'Vertex Pharmaceutical', 'Advanced Micro Devices', 'Honda', 'Carnival Corporations', 'Beyond Petroleum']

        # Post stock ticker too
        listStockTickers = ['FB', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'GS', 'JPM', 'UBS', 'BAC', 'PFE', 'JNJ', 'UBER', 'GM', 'H', 'DAL', 'UAL', 'NVDA', 'TWTR', 'VRTX', 'AMD', 'HMC', 'CCL', 'BP']

        # For now run only for 5 stocks
        listFilenames = ['AppleLastMonth.json', 'MicrosoftLastMonth.json', ]
        listStockNames = ['Apple']
        listFileNames = []
        _sentimentAnalysisBackend, trainedNN, prelabeledTweets, naiveClassifier, vectorDistanceClassifier, discriminantBasedClassifier, adjectiveAdverbPhraseClassifier = self.getTrainedSentimentNN() # here is the call
        listJSONS = []
        index = int(0)
        for filename in listFilenames:
            stockName = listStockNames[index]
            stockTicker = listStockTickers[index]

            bullishP, bearishP, neutralP, sentiment_classification = self.classifyTweetsLastMonth(filename, _sentimentAnalysisBackend, trainedNN, prelabeledTweets, naiveClassifier, vectorDistanceClassifier, discriminantBasedClassifier, adjectiveAdverbPhraseClassifier)
            #Once I have that info just create the appropiate json with those values and post that using the global postToDatabase() method
            #To reset variable for each different filename
            stockBackend = {}
            stockBackend = {
                "stock_name" : str(stockName),
                "stock_ticker": str(stockTicker),
                "sentiment_classification": str(sentiment_classification),
                "bullish_p": str(bullishP),
                "bearish_p": str(bearishP),
                "neutral_p": str(neutralP),

            }
            listJSONS.append(stockBackend)
            index = index + 1
        return listJSONS


    def getQuantBackend(self):

        listJSONQuant = []
        listStockNames = ['Meta', 'Apple', 'Netflix', 'Google', 'Microsoft', 'Amazon', 'Tesla', 'Goldman Sachs', 'JPMorgan', 'UBS', 'Bank of America', 'Pfizer', 'Johnson and Johnson', 'General Motors', 'Hilton', 'Delta', 'United', 'NVIDIA', 'Twitter', 'Vertex Pharmaceutical', 'Advanced Micro Devices', 'Honda', 'Carnival Corporations', 'Beyond Petroleum']
        listStockTickers = ['META', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'GS', 'JPM', 'UBS', 'BAC', 'PFE', 'JNJ', 'GM', 'H', 'DAL', 'UAL', 'NVDA', 'TWTR', 'VRTX', 'AMD', 'HMC', 'CCL', 'BP']

        index = int(0)
        for stock in listStockNames:

            stockTicker = listStockTickers[index]

            #New code to get new quant Backend
            #Name of new file: QuantBackendComplete
            _quant = QuantBackendComplete(stock, stockTicker)
            _quant.getPriceData()
            indicators = _quant.getAllFinancialIndicators()
            finalClassificationQuant = _quant.getFinalClassification(indicators)

            #Process so it turns into an int
            finalClassificationQuant = np.int64(finalClassificationQuant)
            finalClassificationQuant = finalClassificationQuant.item()

            quantBackendCur = {
                "quant_classification": finalClassificationQuant,
                "rsi": str(indicators[0]),
                "ema": str(indicators[1]),
                "macd": str(indicators[2]),
                "obv": str(indicators[3]),
                "rvol": str(indicators[4]),
                "ttm": str(indicators[5]),
                "FamaFrench3FactorMonthlyReturn": str(indicators[6]),
                "FamaFrench3FactorYearlyReturn": str(indicators[7]),
                "FamaFrench5FactorMonthlyReturn": str(indicators[8]),
                "FamaFrench5FactorYearlyReturn": str(indicators[9])

            }

            print(quantBackendCur)
            listJSONQuant.append(quantBackendCur)
            index = index + 1

        return listJSONQuant

    def getFinalClassificationF(self, sentiment_classification, quant_classification):
        #TODO: add a more complex layer of decision making to arrive a final classification
        #Possibly heuristic or pre trained neural network (backtested)

        #Current heuristic: give priority to sentiment over financial metric

        if sentiment_classification == 1 and quant_classification == 1:
            return "Bullish"
        elif sentiment_classification == 1:
            return "Bullish"
        elif sentiment_classification == 2 and quant_classification == 2:
            return "Bearish"
        elif sentiment_classification == 2:
            return "Bearish"
        return "Neutral/Hold"



    def combineBackends(self):

        listJSONQuant = self.getQuantBackend()
        listJSONSentiment = self.getSentimentBackendBERT()

        stocksResults = {}

        index = int(0)
        for stock in listJSONSentiment:

            combinedCur = merge(listJSONQuant[index], listJSONSentiment[index])
            #Append final classification (HC)
            #For now final classification will be either or
            #if quant recommendation is bullish or sentiment recommendation is bullish, then final_recommendation will be bullish
            sentiment_classificationCur = combinedCur["sentiment_classification"]
            quant_classificationCur = combinedCur["quant_classification"]


            stockNameCur = combinedCur["stock_name"]
            final_classificationCur = self.getFinalClassificationF(sentiment_classificationCur, quant_classificationCur)

            print("Stock Name: " + str(stockNameCur))
            print("Category:")
            print(categories[str(stockNameCur)])

            #TODO: fix below quant final classificaiton since currently returning malformed string instead of a 0, 1, 2
            #this is per stock
            #print("Quant Classification: " + quant_classificationCur[2])

            print("Sentiment Classification: " + sentiment_classificationCur)
            finalClassicationOnlySentiment = self.getFinalClassificationSentiment(sentiment_classificationCur)
            print("Final Classification: " + final_classificationCur)

            #TODO: below final classificaiton not working since quant_classificationCur is currently wrong
            #print("Final Classification: " + final_classificationCur)
            print("")
            print("Quant Classification complete: " + str(quant_classificationCur))
            print("Type of Quant classification: " + str(type(quant_classificationCur)))
            print("")
            print("")

            finalClassification = {
            #TODO: add decision code for final classification based on decision being taken in backtasting
            #up for optimizations and modification. Have data to use a ML model if needed.
                #"id": str({stockNameCur} + str(index)),
                "final_classification": str(final_classificationCur),
                "category": categories[str(stockNameCur)],
            }
            combinedCur = merge(combinedCur, finalClassification)

            #Now instead of posting to database, update existing database
            #Need to solve json serializable problem

            #uncomment this
            #putToDatabase(combinedCur)
            print(combinedCur)

            stocksResults[index] = combinedCur
            index = index + 1

        return stocksResults


    def writeResultsToFile(self, jsonResultsStocks):
        with open("initial5.json", "w") as file:
            json.dump(jsonResultsStocks, file, indent=4, sort_keys=True)


    def getFinalClassificationSentiment(self, sentiment_classification):
        #Recall 1 is bullish, 2 is bearish and 0 is neutral
        if sentiment_classification == "1":
            return "Bullish"
        elif sentiment_classification == "2":
            return "Bearish"
        return "Neutral/Hold"


    #TODO: need to incorporate turning recommendation into algorithm both for backtesting
    def getFinalClassification(self, sentiment_classification, quant_classification):
        #TODO:
        #Code algorithm (NN, Voting System?) to get the final_classification of our optimal algorithm that beats the market
        if sentiment_classification == "Bullish" or sentiment_classification == "Bullish\n":
            return "Bullish"
        elif quant_classification == 1:
            return "Bullish"
        elif sentiment_classification == "Bearish" or sentiment_classification == "Bearish\n":
            return "Bearish"
        elif quant_classification == 2:
            return "Bearish"
        return "Neutral/Hold"


if __name__ == "__main__":
    #tweet = sys.argv[1]
    tweet = "trial"
    processNews('AAPL')



    #databaseTryHardCoded()
    #jsonResHardCoded = {
        #"stock_name": "Meta",
        #"sentiment_classification": "Buyy",

    #}
    #putToDatabase(jsonResHardCoded)

    #To run entire backend
    _minervaBackend = MinervaBackend(tweet)
    stockRes = _minervaBackend.combineBackends()


    #To update prices in database
    #listStockTickers = ['META', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'GS', 'JPM', 'UBS', 'BAC', 'PFE', 'JNJ', 'GM', 'H', 'DAL', 'UAL', 'NVDA', 'TWTR', 'VRTX', 'AMD', 'HMC', 'CCL', 'BP']
    #for ticker in listStockTickers:
        #processPrices(ticker)
        #continue
        #processNews(ticker)

    #Below code to upload sample of user favorite stocks to database
    #jsonUserFavorite = {
        #"JavierArevalo" : ["Goldman Sachs", "Facebook", "Apple", "Tesla"]
    #}
    #postToDatabaseFav(jsonUserFavorite)
    print("Done")
