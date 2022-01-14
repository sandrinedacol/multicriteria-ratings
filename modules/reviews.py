from modules.useful_functions import *

from itertools import islice
import json
from textblob import TextBlob
# run python3 -m textblob.download_corpora
from tqdm import tqdm


class Dataset():

    def __init__(self, n_reviews, dataset_name):
        self.n_reviews = n_reviews
        if dataset_name == "Yelp":
            restaurants_ids = load_or_create_cache(n_reviews, "restaurants_ids", self.find_restaurants_id, "dataset/yelp_academic_dataset_business.json")
            self.dict_reviews = load_or_create_cache(n_reviews, "reviews", self.find_restaurants_good_reviews, "dataset/yelp_academic_dataset_review.json", restaurants_ids)
            self.reviews = load_or_create_cache(n_reviews, "sentences", self.extract_texts, self.dict_reviews)

    def find_restaurants_id(self, file_path):
        """
        returns list of 'business_id' with 'Restaurants' in 'categories'
        """
        restaurants_ids = []
        with open(file_path) as f:
            for line in tqdm(f):
                business = json.loads(line)
                if business["categories"]:
                    if 'Restaurants' in business["categories"]:
                        restaurants_ids.append(business["business_id"])
                    
        return restaurants_ids

    def find_restaurants_good_reviews(self, file_path, restaurants_ids):
        """
        returns list of reviews (= dict, so as in json file) matching requirements on business.categories and reviews.helpful
        """
        reviews = []
        if self.n_reviews == 'all':
            with open(file_path) as f:
                for line in tqdm(f):
                    review = json.loads(line)
                    if review["business_id"] in restaurants_ids and review["useful"] > 3:
                        reviews.append(review)
        else:
            with open(file_path) as f:
                reviews = list(islice(f, self.n_reviews))
            reviews = [json.loads(review) for review in reviews]
            reviews = [review for review in reviews if (review["business_id"] in restaurants_ids) and (review["useful"])]
            reviews = [review for review in reviews if review["useful"] > 3]
        print(f'{len(reviews)} reviews')
        return reviews

    def extract_texts(self, dict_reviews):
        """
        returns list of list of strings
        list of reviews x list of raw sentences
        """
        reviews = []
        n_sentences = 0
        for review_idx, dict_review in tqdm(enumerate(dict_reviews)):
            review = Review(dict_review, len(reviews))
            raw_sentences = review.blob.raw_sentences
            # for s in raw_sentences:
            #     if not s.lower().islower():
            #         print(f'\n\n\n !!!!! ISSUE with sentence in review {review_idx} \n {b} \n\n\n')
            for raw_sentence in raw_sentences:
                review.sentences.append(Sentence(raw_sentence, review_idx, n_sentences))
                n_sentences += 1
            reviews.append(review)
        return reviews

class Review():

    def __init__(self, dict_review, review_idx):
        self.idx = review_idx
        self.id = dict_review['review_id']
        self.text = dict_review['text']
        self.blob = TextBlob(self.text)
        self.sentences = []
        self.scores = dict()
        self.labeled_stars = float(dict_review['stars'])
        self.estimated_stars = None
        self.criteria_weights = dict()
        self.n_criteria = 0

    def compute_multicriteria_scores(self, criteria_names):
        extended_scores = {criterion: [] for criterion in criteria_names}
        for sentence in self.sentences:
            if sentence.criterion and sentence.polarity_score:
                criterion, weight = sentence.criterion
                extended_scores[criterion].append((sentence.idx, weight, sentence.polarity_score))
        for criterion in criteria_names:
            if extended_scores[criterion]:
                self.n_criteria += 1
                criterion_weight = sum([weight for (_, weight, _) in extended_scores[criterion]])
                self.criteria_weights[criterion] = criterion_weight
                score = sum([weight * polarity for (_, weight, polarity) in extended_scores[criterion]])/criterion_weight
                self.scores[criterion] = (score+1)/2*5 + 0.5
            else:
                self.scores[criterion] = None
                self.criteria_weights[criterion] = 0
        self.extended_scores = extended_scores

    def show_sentence_criterion_polarity(self):
        text = f'*******************\nREVIEW {self.idx}\n*******************\n'
        for criterion in self.extended_scores.keys():
            text += criterion + '--------------\n'
            weights_sum = sum([weight for _, weight, _ in self.extended_scores[criterion]])
            for sentence_idx, weight, _ in self.extended_scores[criterion]:
                sentence = [sentence for sentence in self.sentences if sentence.idx == sentence_idx][0]
                text += f"{round(weight/weights_sum*100)}% / {sentence.polarity_score} -- {sentence.raw_sentence}\n"
        text += 'None --------------\n'
        for sentence in self.sentences:
            if not sentence.criterion or not sentence.polarity_score:
                text += sentence.raw_sentence + '\n'
        print('\n'+text)
            
    def compute_mean_stars(self, criteria_names, stars_averaging):
        if stars_averaging == 'weighted':
            total_weight = sum(list(self.criteria_weights.values()))
            self.estimated_stars = 0
            for criterion in criteria_names:
                if self.criteria_weights[criterion]:
                    self.estimated_stars += self.criteria_weights[criterion]/total_weight * self.scores[criterion]
        elif stars_averaging == 'equi':
            no_none = [score for score in self.scores.values() if score != None]
            if no_none:
                self.estimated_stars = sum(no_none)/len(no_none)

class Sentence():

    def __init__(self, raw_sentence, review_idx, sentence_idx):
        self.raw_sentence = raw_sentence
        self.review_idx = review_idx
        self.idx = sentence_idx
        self.criterion = None
        self.polarity_score = None
