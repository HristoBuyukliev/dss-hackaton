import re
import pandas as pd 
import nltk
from collections import Counter
import numpy as numpy
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np


train = pd.read_csv("data/dataIMG_train.csv")
test = pd.read_csv("data/dataIMG_evaluation.csv")


train['CleanComment'] = train.Comment.apply(lambda x: re.sub(r'[\.,\?\!:;]', '', x))
train.CleanComment = train.CleanComment.apply(lambda x: x.lower().strip())

stopwords = nltk.corpus.stopwords.words("english")
min_mentions = 10.
min_df = min_mentions/train.shape[0]
print min_df
vectorizer = TfidfVectorizer(stop_words = stopwords, encoding='latin-1', ngram_range=(1,3), min_df=min_df)
corpus = train.CleanComment
X = vectorizer.fit_transform(corpus)
print X.shape[1], ' features selected'

target_cols = [col for col in train.columns if col not in test.columns and col != "CleanComment"]
y = train[target_cols].values

X_train = X[:10000,:]
X_val   = X[10000:, :]
y_train = y[:10000]
y_val   = y[10000:]



def score(y_true, y_pred):
    true_positives = float(np.bitwise_and(y_true, y_pred).sum())
    true_negatives = np.bitwise_and(1-y_true, 1-y_pred).sum()
    false_positives = np.bitwise_and(1-y_true, y_pred).sum()
    false_negatives = np.bitwise_and(y_true, 1-y_pred).sum()

    accuracy  = (true_positives+true_negatives) / y_true.size
    if true_positives + false_positives == 0:
        recall = 0
    else:
        recall    = true_positives / (true_positives + false_negatives)
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    return 0.6*accuracy + 0.25*precision + 0.15*recall 

inputs = Input(shape=(X.shape[1],))
dropout = Dropout(0.5)(inputs)
outputs = Dense(47, activation='sigmoid')(inputs)
model = Model(input=inputs, output=outputs)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])
model.fit(X_train.toarray(), y_train, nb_epoch=20, validation_split = 0.2)

threshold = 0.5
y_pred = model.predict(X_val.toarray())
thresholded_pred = (y_pred > threshold).astype('int64')
print 'threshold: ', threshold
print score(y_val, thresholded_pred)
