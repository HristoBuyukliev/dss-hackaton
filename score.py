import numpy as np 
from keras import backend as K
import tensorflow as tf

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def tensor_score(y_true, y_pred):
    acc = tf.metrics.accuracy(y_true, y_pred)[0]
    pre = tf.metrics.precision(y_true, y_pred)[0]
    rec = tf.metrics.recall(y_true, y_pred)[0]
    print acc[0]
    return 0.6*acc + 0.25*pre + 0.15*rec


def score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).astype('int8')
    y_true = y_true.astype('int8')
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
