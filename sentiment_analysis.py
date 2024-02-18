# Task 21 - Capstone Project

# Create a program that performs sentiment analysis on Amazon product reviews

# Import packages

import spacy
import numpy as np
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

# Import model and data

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

raw_data = pd.read_csv('amazon_product_reviews.csv')

# Clean data by dropping NaN values from the 'reviews.text' column and removing stopwords from the entries (this step takes a while...)

reviews = pd.DataFrame()

reviews['raw_review'] = raw_data['reviews.text'].dropna()

    # This step splits the individual reviews into a list of tokens (if they are not stop words) and then joins them back together as a string

reviews['clean_review'] = reviews['raw_review'].apply(lambda text: ' '.join([token.text.lower() for token in nlp(text) if not token.is_stop]))

# Define 'sentiment' function that returns the polarity of a given review and categorises the review accordingly

def sentiment(text):
    doc = nlp(text)
    polarity = doc._.blob.polarity

    if polarity <= -0.6:
        sentiment = 'Very Negative'
    elif polarity <= -0.2:
        sentiment = 'Negative'
    elif polarity <= 0.2:
        sentiment = 'Neutral'
    elif polarity <= 0.6:
        sentiment = 'Positive'
    else:
        sentiment = 'Very Positive'
    
    return sentiment

# Select 5 random reviews and return their predicted polarity

sample = reviews.sample(5)

for i in range(5):
    print('------------------')
    print(f'Review: {sample.raw_review.iloc[i]}')
    print(f'Sentiment: {sentiment(sample.clean_review.iloc[i])}')

'''
Result:
------------------
Review: What a excellent strean=ming player and with the 4k resolution goes great with my Samsung 4k tv
Sentiment: Very Positive
------------------
Review: Have had it awhile and still learning and adding to it's skills but so far it's pretty good. I'll be adding the additional home automation features soon so we'll see how that goes.
Sentiment: Positive
------------------
Review: Alexa works a lot easier than Siri, Alexa is more like an personal assistant. Works great!!
Sentiment: Positive
------------------
Review: This is an older tablet. I probobly expected too much. If I'm going to deal with lagginess I prefer my ipad2.
Sentiment: Neutral
------------------
Review: Great little streaming box. Better than the fire stick.
Sentiment: Positive
'''