import os
import sys
import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Activation
from keras.layers import Embedding, LSTM
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.callbacks import EarlyStopping
from keras import regularizers, optimizers
from score import score

# BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = "/home/hristo/dss-hackaton"
GLOVE_DIR = BASE_DIR + '/word-embeddings/'
MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 9000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

train = pd.read_csv("data/dataIMG_train.csv")
test = pd.read_csv("data/dataIMG_evaluation.csv")
train['CleanComment'] = train.Comment.apply(lambda x: re.sub(r'[\.,\?\!:;]', '', x))
train.CleanComment = train.CleanComment.apply(lambda x: x.lower().strip())
test['CleanComment'] = test.Comment.apply(lambda x: re.sub(r'[\.,\?\!:;]', '', x))
test.CleanComment = test.CleanComment.apply(lambda x: x.lower().strip())
texts = list(train.CleanComment)
all_texts = list(train.CleanComment) + list(test.Comment)
target_cols = [col for col in train.columns if col not in test.columns]
labels = train[target_cols].values

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in sorted(word_index.items(), key = lambda x: x[1]):
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# class Reweigh(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []

#     def on_epoch_end(self, epochs, logs={}):
#         self.model.sample_weights = np.ones(self.model.training_data.shape[0])

# r = Reweigh()



print('Training model.')
es = EarlyStopping(min_delta=0.01, patience=5)

for penalty in [0.1**i for i in [12,16,20]]:
    print 'penalty:', penalty
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    regularizer = regularizers.l2(penalty)
    lstm1 = Bidirectional(LSTM(100,
                name='first-lstm',
                activation='elu',
                kernel_regularizer=regularizer, 
                return_sequences=True))(embedded_sequences)
    lstm2 = LSTM(47, activation='sigmoid', name='second-lstm')(lstm1)

    loss = 'binary_crossentropy'
    print('loss: ' + loss)
    model = Model(sequence_input, lstm2)
    model.compile(loss=loss,
                  optimizer='rmsprop',
                  metrics=['acc'])


    model.fit(x_train, y_train, epochs=100, 
            validation_data=(x_val, y_val), 
            batch_size=128, 
            callbacks=[es],
            verbose=1)

    y_pred = model.predict(x_val)
    print(y_pred)
    print(score(y_val, y_pred))
    thresholds = np.linspace(0,1,11)
    for threshold in thresholds:
        print(threshold)
        print(score(y_val, y_pred, threshold))

