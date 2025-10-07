import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import ssl

# ---- LOAD DATA ----
df = pd.read_csv("WELFake_Dataset.csv")
df = df.dropna()

x = df.drop(columns=['Unnamed: 0','label'])
y = df.label
voc_size = 5000

ps = PorterStemmer()
corpus = []
for i in range(len(x)):
    review = re.sub('[^a-zA-Z]', ' ', x['title'].iloc[i])  # use .iloc for row access
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

onehot_corp=[one_hot(words, voc_size) for words in corpus]
sent_len = 20
padded_corp = pad_sequences(onehot_corp, padding = 'pre', maxlen = sent_len)
print(padded_corp)

Embedding_vector_features = 40
classifier = Sequential()
classifier.add(Embedding(voc_size, Embedding_vector_features, input_length = sent_len))
classifier.add(LSTM(100))
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(classifier.summary())