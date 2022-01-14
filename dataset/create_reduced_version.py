import json

n_first = 10

full_dataset = f'yelp_academic_dataset_review'
reduced_dataset = f'{full_dataset}_{n_first}'

reviews = []

with open(f'{full_dataset}.json') as full:
    full_reviews = json.load(full)
    reviews = full_reviews[:n_first]
with open(f'{reduced_dataset}.txt', 'w') as reduced:
    reduced.write(json.dumps(reviews, indent = 4))


