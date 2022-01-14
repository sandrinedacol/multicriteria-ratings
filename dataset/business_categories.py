import json

table = f'yelp_academic_dataset_business.json'
categories_file = f'business_categories.txt'
categories = []
with open(table) as f:
    for line in f:
        business = json.loads(line)
        if business["categories"]:
            cats = business["categories"].split(', ')
            categories += cats
categories = set(categories)
with open(categories_file, 'w') as f:
    for cat in categories:
        f.write(cat + '\n')
