# -*- coding: utf-8 -*-
"""sma_5_sentiment_prac_exam.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UYA7O8CTQNQe3U2B6bVjK8Bi2PZCjGpJ
"""

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from CSV
data = pd.read_csv('sentiment.csv')

# Perform sentiment analysis using TextBlob
data['polarity'] = data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Categorize tweets into positive, negative, and neutral based on polarity
data['sentiment'] = data['polarity'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))

# print(data)
# Count the number of positive, negative, and neutral tweets
sentiment_counts = data['sentiment'].value_counts()

# Plotting the sentiment analysis
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Analysis')
plt.show()

# Separate positive and negative tweets
positive_tweets = data[data['sentiment'] == 'positive']
negative_tweets = data[data['sentiment'] == 'negative']
neutral_tweets = data[data['sentiment'] == 'neutral']

# Print a few positive tweets
print("Positive Tweets:")
for tweet in positive_tweets['text'].head():
    print("-", tweet)

# Print a few negative tweets
print("\nNegative Tweets:")
for tweet in negative_tweets['text'].head():
    print("-", tweet)

# Print a few neutral tweets
print("\nNegative Tweets:")
for tweet in neutral_tweets['text'].head():
    print("-", tweet)


# Plot the distribution of polarity scores for positive and negative tweets
# A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset, analogous to a histogram.
sns.kdeplot(positive_tweets["polarity"], shade=True, label="Positive")
sns.kdeplot(negative_tweets["polarity"], shade=True, label="Negative")
sns.kdeplot(neutral_tweets["polarity"], shade=True, label="Neutral")
plt.xlabel("Polarity Score")
plt.ylabel("Density")
plt.title("Sentiment Analysis of Tweets")
plt.legend()
plt.show()