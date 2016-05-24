import json
import pandas as pd

review_fileName = '../Data/yelp_academic_dataset_review.json'
business_fileName = '../Data/yelp_academic_dataset_business.json'
user_fileName = '../Data/yelp_academic_dataset_user.json'

with open(review_fileName) as f:
    reviews = pd.DataFrame(json.loads(line) for line in f)
f.close()
reviews = reviews.set_index(['review_id'])
reviews = reviews[['business_id', 'stars', 'text', 'user_id']]

with open(business_fileName) as f:
	business = pd.DataFrame(json.loads(line) for line in f)
f.close()
business = business.set_index(['business_id'])


import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

sentence = reviews[:1]['text'].values
words = nltk.word_tokenize(sentence)

filtered_words = [w for w in words if w not in stopwords.words('english')]
stemmer = SnowballStemmer('english')
stemmed_words = [stemmer.stem(w) for w in filtered_words]

def process_text(sentence, stemmer=SnowballStemmer('english'), stopwords=set(stopwords.words('english'))):
	return [stemmer.stem(w) for w in nltk.word_tokenize(sentence.lower()) if w.isalpha() and w not in stopwords]