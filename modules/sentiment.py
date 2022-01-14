from modules.useful_functions import *

import math
import matplotlib.pyplot as plt
# nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
# run python3 -m textblob.download_corpora
from flair.models import TextClassifier
from flair.data import Sentence as Flair_Sentence
from tqdm import tqdm


class Sentiment():

    def __init__(self, n_reviews, sentiment_analyzer):
        self.n_reviews = n_reviews
        self.sentiment_analyzer = sentiment_analyzer
        if self.sentiment_analyzer == 'NLTK':
            self.nltk = SentimentIntensityAnalyzer()
        elif self.sentiment_analyzer == 'Flair':
            self.flair = TextClassifier.load('en-sentiment')
        
    def compute_polarity_score(self, raw_sentence):
        if self.sentiment_analyzer == 'NLTK':
            polarity_score = self.compute_NLTK(raw_sentence)
        elif self.sentiment_analyzer == 'TextBlob':
            polarity_score = self.compute_TextBlob(raw_sentence)
        elif self.sentiment_analyzer == 'Flair':
            polarity_score = self.compute_Flair(raw_sentence)
        return polarity_score

    def compute_NLTK(self, raw_sentence):
        ps = self.nltk.polarity_scores(raw_sentence)
        return ps['compound']

    def compute_TextBlob(self, raw_sentence):
        sentence = TextBlob(raw_sentence)
        return sentence.sentiment.polarity

    def compute_Flair(self, raw_sentence):
        sentence = Flair_Sentence(raw_sentence)
        self.flair.predict(sentence)
        polarity, score = sentence.labels[0].value, sentence.labels[0].score
        find_sign = lambda string: 1 if string == 'POSITIVE' else -1
        return find_sign(polarity) * score

    def compute_raw_polarities(self, corpus):
        self.raw_polarities = []
        print('Computing raw polarities...')
        for sentence in tqdm(corpus):
            polarity_score = self.compute_polarity_score(sentence)
            self.raw_polarities.append(polarity_score)

    def show_polarity_distribution(self, alpha):
        name = f"fig/{self.n_reviews}/{self.sentiment_analyzer}"
        try:
            plt.hist([pol for pol in self.polarities if pol], bins=20)
            name += f'_processed_polarities_alpha-{alpha}.svg'
        except AttributeError:
            plt.hist(self.raw_polarities, bins=20)
            name += '_raw_polarities.svg'
        plt.xlabel('polarity')
        plt.ylabel('count')
        plt.legend()
        plt.savefig(name)
        plt.clf()
        
    def process_polarities(self, alpha):
        tanh = lambda pol: math.tanh(alpha*pol) if pol else None
        self.polarities = [tanh(polarity) for polarity in self.raw_polarities]

