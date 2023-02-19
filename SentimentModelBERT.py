from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import numpy as np
from scipy.special import softmax


#this class will make a sentiment classification for one tweet/news at a time 
class SentimentModelBERT:

	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
		self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

	def makeRecommendation(self, news):

		tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
		model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
		config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

		# Labels:
		# 0: negative 
		# 1: Neutral
		# 2: positive 
		processedText = self.preprocess(self, news)
		encoded_input = tokenizer(processedText, return_tensors='pt')
		output = model(**encoded_input)
		scores = output[0][0].detach().numpy()
		scores = softmax(scores)
		print(scores)
		print("")

		#return array with [negativeScore, neutralScore, positiveScore]
		ranking = np.argsort(scores)
		ranking = ranking[::-1]
		finalScores = {}
		for i in range(scores.shape[0]):
			label = config.id2label[ranking[i]]
			score = scores[ranking[i]]
			finalScores[str(label)] = score

		return finalScores

	def preprocess(self, text):
		new_text = []
		for t in text.split(" "):
			t = '@user' if t.startswith('@') and len(t) > 1 else t
			t = 'http' if t.startswith('http') else t
			new_text.append(t)
		return " ".join(new_text)


#try with main
if __name__ == "__main__":
	trialNews = "Goldman stock will rise"
	_sentimentModel = SentimentModelBERT
	recommendation = _sentimentModel.makeRecommendation(_sentimentModel, trialNews)
	print("Sentence: ")
	print(trialNews)
	print(recommendation)
	#negativeRating = recommendation[0]
	#neutralRating = recommendation[1]
	#positiveRating = recommendation[2]
	#print("Negative rating: " + str(negativeRating))
	#print("Neutral rating: " + str(neutralRating))
	#print("Positive Rating: " + str(positiveRating))


