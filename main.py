from modules.reviews import *
from modules.criteria import *
from modules.sentiment import *
from modules.parameters import *

import yaml
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class MultiCriteriaScoresExtractor():

    def __init__(self):
        with open(f'parameters.yaml', 'r') as cfile:
            yml = yaml.load(cfile, Loader=yaml.FullLoader)
            self.args = Parameters(yml)
        self.theta, self.alpha = self.args.theta, self.args.alpha
        self.dataset = Dataset(self.args.n_reviews, self.args.dataset_name)
        self.criteria = Criteria(self.args.n_reviews, self.args.criteria)
        self.sentiment = Sentiment(self.args.n_reviews, self.args.sentiment_analyzer)

    def first_step(self):
        corpus = [s.raw_sentence for review in self.dataset.reviews for s in review.sentences]
        self.criteria.compute_raw_similarities(corpus, self.args.metric)
        self.sentiment.compute_raw_polarities(corpus)

    def second_step(self):
        sentences_criteria = self.criteria.process_similarities(self.theta)
        self.sentiment.process_polarities(self.alpha)
        if self.args.mode == 'run':
            print('\nProcessing similarities and polarities...')
        for review in tqdm(self.dataset.reviews):
            for sentence in review.sentences:
                sentence.criterion = sentences_criteria[sentence.idx]
                sentence.polarity_score = self.sentiment.polarities[sentence.idx]
            review.compute_multicriteria_scores(self.criteria.names)
            review.compute_mean_stars(self.criteria.names, self.args.stars_averaging)
        
        

    def run(self):
        self.first_step()
        self.criteria.show_similarities(self.theta)
        self.criteria.show_weights()
        self.sentiment.show_polarity_distribution(self.alpha)
        self.show_example_1('xwNlRAyJ1tLQQcN8IQmenA')
        self.second_step()
        self.criteria.visualize_criteria()
        self.criteria.show_similarities(self.theta)
        self.criteria.show_weights()
        self.sentiment.show_polarity_distribution(self.alpha)
        self.show_error_distribution()
        self.show_example_2('xwNlRAyJ1tLQQcN8IQmenA')
        self.write_result_file()
        
    def show_example_1(self, review_id):
        for review in self.dataset.reviews[:1000]:
            if review.id == review_id:
                print(f'\n*******************\nExample: review {review_id}\n*******************')
                for sentence in review.sentences:
                    print(sentence.raw_sentence)
                    for i, similarity in enumerate(self.criteria.raw_similarities):
                        if i == sentence.idx:
                            print(similarity)

    def show_error_distribution(self):
        errors = []
        for review in self.dataset.reviews:
            errors.append(review.estimated_stars - review.labeled_stars)
        plt.hist(errors, bins=20)
        plt.xlabel('error (stars)')
        plt.ylabel('count')
        plt.savefig(f"fig/{self.args.n_reviews}/error_distribution.svg")
        plt.clf()

    def show_example_2(self, review_id):
        for review in self.dataset.reviews[:1000]:
            if review.id == review_id:
                review.show_sentence_criterion_polarity()

    def write_result_file(self):
        result_file = ""
        for review in self.dataset.reviews:
            result_file += str(review.scores) + f"\nlabel: {review.labeled_stars} / result: {round(review.estimated_stars)}  \n" + review.text + '\n******************************\n'
        with open(f'output/{self.args.dataset_name}_{self.args.n_reviews}.txt', 'w') as f:
            f.write(result_file)
        
        result = []
        discarded_reviews = 0
        kept_reviews = 0
        for review in self.dataset.reviews:
            dict_review = self.dataset.dict_reviews[review.idx]
            if abs(review.labeled_stars - review.estimated_stars) <= self.args.error_max:
                kept_reviews += 1
                review_scores = dict()
                for criterion, score in review.scores.items():
                    if score != None:
                        review_scores[criterion.lower()] = round(review.scores[criterion])
                dict_review['multicriteria_stars'] = review_scores
                result.append(dict_review)
            else:
                discarded_reviews += 1
        with open(f"output/yelp_academic_dataset_review_multiscored_{self.args.n_reviews}.json", 'w') as f:
            json.dump(result, f)
        print(f"\ndiscarded reviews: {discarded_reviews} (on {kept_reviews + discarded_reviews}, {100*discarded_reviews/(kept_reviews + discarded_reviews)}%)\n")


    def test(self):
        # initiate
        self.first_step()
        dataset_copy = copy.deepcopy(self.dataset)
        criteria_copy = copy.deepcopy(self.criteria)
        self.losses = []
        self.efficiencies = []
        self.estimated_ratios = []
        self.means = []
        self.variances = []
        # define values for parameter
        val_func = lambda val: val/self.args.n_values*(self.args.max_value - self.args.min_value) + self.args.min_value
        values = [round(val_func(val), self.args.significant_digits -1) for val in list(range(self.args.n_values))]
        values = sorted(list(set(values)))
        print('\n' + self.args.tested_parameter)
        for value in values:
            setattr(self, self.args.tested_parameter, value)
            print(value)
            self.dataset = copy.deepcopy(dataset_copy)
            self.criteria = copy.deepcopy(criteria_copy)
            self.second_step()
            self.evaluate_quality()
        self.plot_param_influence(self.args.tested_parameter, values)
        
    def evaluate_quality(self, ):
        error_list = []
        number_of_estimated_scores = 0
        for review in self.dataset.reviews:
            if review.estimated_stars:
                if self.args.loss_type == 'MAE':
                    error_list.append(abs(review.estimated_stars - review.labeled_stars))
                elif self.args.loss_type == 'rMSE':
                    error_list.append((review.estimated_stars - review.labeled_stars)**2)
            number_of_estimated_scores += review.n_criteria
        loss = sum(error_list)/len(error_list)
        if self.args.loss_type == 'rMSE':
            loss = math.sqrt(loss)
        self.losses.append(loss)
        error_variance = np.var(error_list)
        self.variances.append(error_variance)
        estimated_ratio  = number_of_estimated_scores/(len(self.dataset.reviews)*len(self.args.criteria))
        self.estimated_ratios.append(estimated_ratio)
        print(f'{self.args.loss_type} loss: {loss} / error variance: {error_variance} / estimated scores ratio: {estimated_ratio}')
        file_path = f"tests/tests.csv"
        file_content = f"\n{self.args.dataset_name},{self.args.n_reviews},{self.args.criteria}, {self.args.metric}, {self.args.sentiment_analyzer}, {self.args.loss_type}, {self.args.stars_averaging}, {self.theta}, {self.alpha}, {loss}, {error_variance}"
        if os.path.exists(file_path):
            with open(file_path, 'a') as f:
                f.write(file_content)
        else:
            file_content = "dataset_name,n_reviews,criteria,metric,sentiment_analyzer,loss_type,stars_averaging,theta,alpha,loss,error_variance" + file_content
            with open(file_path, 'a') as f:
                f.write(file_content)

    def plot_param_influence(self, parameter, param_values):
        # loss
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel(parameter)
        ax1.set_ylabel(f'{self.args.loss_type} loss (stars)', color=color)
        ax1.plot(param_values, self.losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color) 
        # variance of error
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('variance (stars^2)', color=color)
        ax2.plot(param_values, self.variances, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        # plot
        name = f"tests/fig/{self.args.n_reviews}/{parameter}_"
        if parameter == 'alpha':
            plt.savefig(f"{name}_theta{self.theta}.svg")
        elif parameter == 'theta':
            plt.savefig(f"{name}_alpha{self.alpha}.svg")
        plt.clf()


def main():
    pipeline = MultiCriteriaScoresExtractor()
    if pipeline.args.mode == 'run':
        pipeline.run()
    elif pipeline.args.mode == 'test':
        pipeline.test()

if __name__ == "__main__":
    main()
    

