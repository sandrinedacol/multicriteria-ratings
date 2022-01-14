from modules.useful_functions import *

from itertools import compress
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

class Criteria():

    def __init__(self, n_reviews, criteria):
        self.n_reviews = n_reviews
        tuples = [t for t in criteria.items()]
        self.names = [t[0] for t in tuples]
        self.lexicon = [t[1] for t in tuples]

    def compute_raw_similarities(self, corpus, metric):
        self.corpus = corpus + self.lexicon
        self.embeddings = load_or_create_cache(self.n_reviews, 'embeddings', self.convert_doc_to_numeric_data)
        sentence_embeddings = self.embeddings[:-len(self.names)]
        criteria_embeddings = self.embeddings[-len(self.names):]
        if metric == 'euclidian':
            euclidian_dists = np.array(cdist(sentence_embeddings, criteria_embeddings, metric='euclidean'))
            to_sim = np.vectorize(lambda dist: 1/(1+dist))
            self.raw_similarities = to_sim(euclidian_dists)
        elif metric == 'cosine':
            cosine_sims = np.array(cosine_similarity(sentence_embeddings, criteria_embeddings))
            to_sim = np.vectorize(lambda sim: (sim + 1)/2)
            self.raw_similarities = to_sim(cosine_sims)
        
    def convert_doc_to_numeric_data(self):
        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings = model.encode(self.corpus, show_progress_bar=True)
        return embeddings

    def show_similarities(self, theta):
        name = f"fig/{self.n_reviews}/"
        try:
            similarities = self.similarities
            name += f'processed_similarities_theta-{theta}.svg'
            sims = similarities.T.tolist()
            for i, sim in enumerate(sims):
                sims[i] = list(filter(lambda a: a != 0, sim))
        except AttributeError:
            similarities = self.raw_similarities
            name += 'raw_similarities.svg'
            sims = similarities.T.tolist()
        plt.hist(sims, bins=20, label=self.names)
        plt.xlabel('similarity to criterion')
        plt.ylabel('count')
        plt.legend()
        plt.savefig(name)
        plt.clf()

    def show_weights(self):
        try:
            similarities = self.similarities
        except AttributeError:
            similarities = self.raw_similarities
        weights = dict()
        min_weight = 10000
        for i, criterion in enumerate(self.names):
            weight = np.mean(similarities.T[i])
            weights[criterion] = weight
            min_weight = min([min_weight, weight])
        for criterion in self.names:
            weights[criterion] = weights[criterion]/min_weight
        print(f'\naveraged weights:\n{weights}\n')

    def process_similarities(self, theta):
        quantile = np.quantile(self.raw_similarities, theta)
        sentences_criteria = []
        similarities = []
        for similarity in self.raw_similarities:
            weight = max(similarity)
            sim = [0,0,0,0]
            if weight >= quantile:
                criterion_idx = similarity.tolist().index(weight)
                criterion = self.names[criterion_idx]
                sentences_criteria.append((criterion, weight-quantile))
                sim[criterion_idx] = weight-quantile
            else:
                sentences_criteria.append(None)
            similarities.append(sim)
        self.similarities = np.array(similarities)
        return sentences_criteria

    def reduce_embeddings_dimensionality(self, n_components, min_dist=0.1):
        umap_embeddings = umap.UMAP(n_neighbors=15, n_components=n_components, min_dist=min_dist, metric="cosine").fit_transform(self.embeddings)
        return umap_embeddings

    def visualize_criteria(self):
        umap_data = load_or_create_cache(self.n_reviews, '2D-projection_embeddings', self.reduce_embeddings_dimensionality, 2, 0.0)
        x, y = umap_data.T[0], umap_data.T[1]
        # outliers_idx = [x.tolist().index(i) for i in x if i > 7]
        all_zeros = [not s.any() for s in self.similarities]
        x_, y_ = list(compress(x, all_zeros)), list(compress(y, all_zeros))
        plt.scatter(x_, y_, color='#BDBDBD', alpha=0.5, s=5, linewidths=0)
        for i, criterion in enumerate(self.names):
            sim_max = self.similarities.T[i].max()
            sim_min = self.similarities.T[i].min()
            reduce = lambda sim: ((sim-sim_min)/(sim_max-sim_min))
            alpha = reduce(self.similarities.T[i])
            plt.scatter(x, y, label=criterion, alpha=alpha, s=7, linewidths=0)
        leg = plt.legend()
        leg.markerscale = 1
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
            lh._sizes = np.array([90])
            plt.xlim([4,15])
            plt.ylim([4,13])
        plt.savefig(f'fig/{self.n_reviews}/2D_embeddings.png', dpi=200)
        plt.clf()

