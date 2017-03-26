import re
import pandas as pd 
import nltk
import seaborn as sns
from collections import Counter
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

train = pd.read_csv("data/dataIMG_train.csv")
test = pd.read_csv("data/dataIMG_evaluation.csv")

train_country = pd.get_dummies(train.Country).values[:10000]
val_country   = pd.get_dummies(train.Country).values[10000:]
test_country  = pd.get_dummies(test.Country).values
train['Part'] = train["Unique.Line.ID"].apply(lambda x: int(x[-1]))
test['Part']  = test["Unique.Line.ID"].apply(lambda x: int(x[-1]))



train['CleanComment'] = train.Comment.apply(lambda x: re.sub(r'[\.,\?\!:;]', '', x))
train.CleanComment = train.CleanComment.apply(lambda x: x.lower().strip())
test['CleanComment'] = test.Comment.apply(lambda x: re.sub(r'[\.,\?\!:;]', '', x))
test.CleanComment = test.CleanComment.apply(lambda x: x.lower().strip())

stopwords = nltk.corpus.stopwords.words("english")
min_mentions = 10.
min_df = min_mentions/train.shape[0]
vectorizer = TfidfVectorizer(
        stop_words = stopwords, 
        encoding='latin-1', 
        ngram_range=(1,5), 
        min_df=min_df)
corpus = train.CleanComment
X = vectorizer.fit_transform(corpus)
print X.shape[1], ' features selected'

target_cols = [col for col in train.columns if col not in test.columns and col != "CleanComment"]
y = train[target_cols].values

train_size = X.shape[0]

X_train = X[:train_size,:]
X_val   = X[train_size:, :]
y_train = y[:train_size]
y_val   = y[train_size:]
y_test = np.zeros((test.shape[0],47))
X_test = vectorizer.transform(test.CleanComment)


def score(y_true, y_pred):
    threshold = 0.5
    y_pred = (y_pred > threshold).astype('int64')
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

# def clean()
# counter = 0
# for i in range(10000,14618):
#    if "none" in train.loc[i]["CleanComment"][-5:]:
#        y_pred[i-10000,23:-1]=0
#        counter+=1
#    if "none" in train.loc[i]["CleanComment"][:5]:
#        y_pred[i-10000,:23]=0
#        counter+=1
# print(counter)

# tsne = TSNE(n_components=2, metric="cosine")
# reduced = tsne.fit_transform(X.toarray()[:1000])
# sns.jointplot(reduced[:,0], reduced[:,1])
# sns.plt.show()

# lda = LatentDirichletAllocation(n_topics=200)
# lda.fit(X)
# X_lda = lda.transform(X_train)

# tsne = TSNE(n_components=2, metric="cosine")
# reduced = tsne.fit_transform(X_lda[:1000])
# sns.jointplot(reduced[:,0], reduced[:,1])
# sns.plt.show()

# inputs = Input(shape=(X_train.shape[1],))
# dropout = Dropout(0.2)(inputs)
# mid_layer = Dense(300, activation='relu')(dropout)
# dropout2 = Dropout(0.2)(mid_layer)
# mid_layer2 = Dense(300, activation='relu')(dropout2)

# outputs = Dense(47, activation='sigmoid')(inputs)
# model = Model(input=inputs, output=outputs)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy')
# model.fit(X_train.toarray(), y_train, nb_epoch=100, validation_split = 0.2)

# threshold = 0.5
# y_pred = model.predict(lda.transform(X_val))
# thresholded_pred = (y_pred > threshold).astype('int64')
# print 'threshold: ', threshold
# print score(y_val, thresholded_pred)

# logistic regression
y_pred = np.zeros(y_test.shape)
for col in range(47):
    lr = LogisticRegression(penalty='l1', C=2)
    print "fitting..."
    lr.fit(X_train, y_train[:, col])
    print "predicting..."
    y_pred[:,col] = lr.predict_proba(X_test)[:,1]

y_test_df = pd.DataFrame(data=y_pred, columns = target_cols)
output = pd.concat(test, y_test_df)
output.to_csv(output.csv)

#cosine similarity classification
# y_pred = np.zeros(y_val.shape)
# for col in range(47):
#     knn = KNeighborsClassifier(n_neighbors=20, weights="distance", leaf_size=100)
#     print "fitting..."
#     knn.fit(X_train, y_train[:, col])
#     print X_train.shape, y_train[:, col].shape
#     print "predicting..."
#     y_pred[:,col] = knn.predict_proba(X_val)[:,1]



# #svm's
# y_pred = np.zeros(y_val.shape)
# for col in range(1):
#     clf = SVC(probability=True)
#     print "fitting..."
#     clf.fit(X_train, y_train[:, col])
#     print "predicting..."
#     y_pred[:,col] = clf.predict_proba(X_val)[:,1]


