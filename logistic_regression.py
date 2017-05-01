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
from keras.layers import Input, Dense, Dropout, concatenate
from keras.models import Model
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import LatentDirichletAllocation
from score import score
import gc


TRAIN_SIZE         = 10000
VAL_SIZE           = 14618 - TRAIN_SIZE
LDA_TOPICS         = 50
NGRAM_MAX_LEN      = 5
MIN_NGRAM_MENTIONS = 15


train = pd.read_csv("data/dataIMG_train.csv")
test = pd.read_csv("data/dataIMG_evaluation.csv")
target_cols = [col for col in train.columns if col not in test.columns and col != "CleanComment"]

test.index += train.shape[0]


# remove punctuation and change to lowercase
train['CleanComment'] = train.Comment.apply(lambda x: re.sub(r'[\.,\?\!:;]', '', x))
train.CleanComment = train.CleanComment.apply(lambda x: x.lower().strip())
test['CleanComment'] = test.Comment.apply(lambda x: re.sub(r'[\.,\?\!:;]', '', x))
test.CleanComment = test.CleanComment.apply(lambda x: x.lower().strip())

all_data = pd.concat([train, test])

# convert the cleaned comment to BOW
stopwords = nltk.corpus.stopwords.words("english")
vectorizer = TfidfVectorizer(
        stop_words = stopwords, 
        encoding='latin-1', 
        ngram_range=(1,NGRAM_MAX_LEN), 
        min_df=MIN_NGRAM_MENTIONS)
corpus = all_data.CleanComment
X = vectorizer.fit_transform(corpus)
print X.shape[1], ' features selected'

# add LDA columns
lda_feature_names = ['LDA-'+str(i) for i in range(LDA_TOPICS)]
lda = LatentDirichletAllocation(n_topics=LDA_TOPICS)
lda.fit(X)
X_lda = lda.transform(X)
lda_dict = dict(zip(lda_feature_names, X_lda.T))
all_data = all_data.assign(**lda_dict)

# add handcrafted features
def extract_comment_part(line_id):
    if '.' in line_id:
        return int(re.split(r'\.|-', line_id)[-1])
    else:
        return 1

def extract_comment_number(line_id):
    if '-' in line_id:
        return int(re.split(r'\.|-', line_id)[-2])
    else:
        return 1
        # return int(re.search(r'\d+', line_id.split('.')[0]).group())
all_data['Part'] = all_data["Unique.Line.ID"].apply(extract_comment_part)
all_data['CommentNumber'] = all_data["Unique.Line.ID"].apply(extract_comment_number)

def starting_str(line_id):
    return re.search(r'^[a-zA-Z]+', line_id).group().lower()
all_data['starting_str'] = all_data["Unique.Line.ID"].apply(starting_str)
all_data = pd.get_dummies(all_data, columns = ['starting_str', 'Country'])

end_none_f = lambda x: int(bool(re.search(r'none$', x)))
all_data['ends_with_none'] = all_data.CleanComment.apply(end_none_f)
begin_none_f = lambda x: int(bool(re.search(r'^none', x)))
all_data['starts_with_none'] = all_data.CleanComment.apply(begin_none_f)

non_numeric_columns = ['Comment', 'CleanComment', 'Unique.Line.ID']
basis_cols = [col for col in all_data.columns if col not in target_cols+non_numeric_columns]

print len(basis_cols), ' non-word cols selected'
X_all = np.hstack((all_data[basis_cols].values, X.toarray()))
# X_all = X.toarray()
# X_all = all_data[basis_cols].values

X_train = X_all[:TRAIN_SIZE,:]
X_val   = X_all[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE,:]
X_test  = X_all[TRAIN_SIZE+VAL_SIZE:,:]
y_train = all_data.ix[:TRAIN_SIZE-1, target_cols].values
y_val   = all_data.ix[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE-1, target_cols].values

normalizer = Normalizer()
normalizer.fit(X_train)
X_train = normalizer.transform(X_train)
X_val   = normalizer.transform(X_val)
X_test  = normalizer.transform(X_test)

# possible_penalties = [0.1**i for i in range(4,7)]
# for penalty in possible_penalties:
#     for loss in ['mse', 'mae', 'binary_crossentropy', 'msle', 'kld', 'hinge', 'squared_hinge', 'cosine_proximity']:
#         print loss, penalty
#         inputs = Input(shape=(X_train.shape[1],))

#         regularizer = regularizers.l1(penalty)
#         # mid_layer1 = Dense(300, activation='relu', 
#         #                   W_regularizer=regularizer
#         #                   )(inputs)

#         # # regularizer2 = regularizers.l2(beta_penalty)
#         # mid_layer2 = Dense(300, activation='relu', 
#         #                    W_regularizer=regularizer
#         #                    )(mid_layer1)

#         outputs = Dense(47, activation='sigmoid', W_regularizer=regularizer)(inputs)
#         model = Model(input=inputs, output=outputs)
#         model.compile(optimizer='rmsprop',
#                       loss='binary_crossentropy')
#         model.fit(X_train.toarray(), y_train, nb_epoch=10, validation_split = 0.0, verbose=0)

#         y_pred = model.predict(X_val.toarray())
#         thresholds = np.linspace(0,1,11)
#         print(max([score(y_val, y_pred, threshold) for threshold in thresholds]))
# threshold = 0.5
# y_pred = model.predict(lda.transform(X_val))
# thresholded_pred = (y_pred > threshold).astype('int64')
# print 'threshold: ', threshold
# print score(y_val, thresholded_pred)

# # logistic regression
# for c in [0.01, 0.03, 0.06, 0.1, 0.15, 0.21, 0.4, 0.7, 1, 1.3, 1.6, 2, 3, 4]:
#     print c
#     y_pred = np.zeros(y_val.shape)

#     for col in range(47):
#         lr = LogisticRegression(penalty='l1', C=c)
#         lr.fit(X_train, y_train[:, col])
#         y_pred[:,col] = lr.predict_proba(X_val)[:,1]
thresholds = np.linspace(0,1,11)
print(max([score(y_val, y_pred, threshold) for threshold in thresholds]))

# neural networks
basis_dims = len(basis_cols)
possible_penalties = [0.1**i for i in range(3,8)[::-1]]
for dense_penalty in possible_penalties:
    for sparse_penalty in possible_penalties:
        print dense_penalty, sparse_penalty
        y_pred = np.zeros(y_val.shape)

        inputs_dense = Input(shape=(basis_dims,))
        inputs_sparse = Input(shape=(X.shape[1],))
        dense_regularizer = regularizers.l2(dense_penalty)
        sparse_regularizer = regularizers.l2(sparse_penalty)

        outputs_d = Dense(20, activation='sigmoid', kernel_regularizer=dense_regularizer)(inputs_dense)
        outputs_s = Dense(200, activation='sigmoid', kernel_regularizer=sparse_regularizer)(inputs_sparse)

        merge = concatenate([outputs_s, outputs_d])
        outputs = Dense(47, activation='sigmoid')(merge)
        model = Model(input=[inputs_dense, inputs_sparse], output=outputs)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy')
        model.fit([X_train[:,:basis_dims], X_train[:,basis_dims:]], y_train, nb_epoch=10)
        y_pred = model.predict([X_val[:, :basis_dims], X_val[:, basis_dims:]])
        thresholds = np.linspace(0,1,11)
        print(max([score(y_val, y_pred, threshold) for threshold in thresholds]))




# y_test_df = pd.DataFrame(data=y_pred, columns = target_cols)
# output = pd.concat(test, y_test_df)
# output.to_csv(output.csv)

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


