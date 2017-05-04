# dss-hackaton
Our (team puff-puff-overfit) solution for the Data Science Society hackaton.

## Data cleaning
We used pretty standard data cleaning techniques:
- convert the categorical variables (in this case country) to dummy
- remove punctuation
- remove stopwords (with nltk)
- select the n-grams with more than 10 mentions, where n <= 5
- weigh by tf-idf

## Feature engineering
###### From the text
We tried LDA, PCA, t-SNE, and combinations of those (e.g. LDA to 1000 topics and then t-SNE). None of them seemed to work particularly well, even though LDA -> tSNE gave some pretty cool visualizations that clearly contained a lot of structure

###### Additional
We created a dummy variable for country, from the region (contained in the id), and whether the text contained, started or ended with "none". Also, the comment number and part.

## Approaches
We tried several easy (in the sense of easy to implement and fast to compute) methods:
- Logistic regression
- SVMs, with polynomial, linear, and rbf kernel
- Distance based algorithms, where the distance was cosine similarity (which works well with text)
- Decision tree-based algorithms
- As an extension to the logistic regression, a multilayer perceptron, implemented in keras
All of those capped at around 0.83 score. For some reason, the implementation of logistic regression in keras gave significantly lower score than the equivalent implementation in scikit-learn. If that bug is found, it may improve the more complicated models we tried in keras. 

Additionally, we played with some deep learning. Unfortunately, it required a lot of hyperparameter optimization, architecture exploration, and so on, so we didn't have time to fully explore the possibilities. 
We finally (after the datathon ended) settled on using pre-trained word embeddings, trained on 400k words from Wikipedia (see [here](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) and [here](https://nlp.stanford.edu/projects/glove/)). We used these weight as frozen, and then added two bidirectional LSTM layers, with 100 and 47 output units, and ELU and sigmoid activations. The running time was around 10 minutes, which was pretty annoying to wait through. If you have more computational power, we think that improvements can be made.
The best score, when using only the raw text, and none of the additional features, was 0.838, which is still somewhat better than logistic regression. The model optimizes binary crossentropy (since the target metric is non-differentiable), and then chooses the optimal threshold from the interval 0-1. This definitely feels unsatisfying, but ultimately, the metric is unsatisfying so we don't feel too bad about this.

## Future work
- Fixing the keras bug
- We didn't use ensembles, which should significantly improve the score
- Increasing the depth of the neural network
- The data is definitely not enough for more complicated models, so finding additional (even somewhat unrelated) data would be useful
- Adding the additional features will probably improve performance a bit.