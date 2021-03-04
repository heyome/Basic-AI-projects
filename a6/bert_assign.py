# HW6 - "step 0" following Jay Alammar's tutorial, 
# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# including some code from there

import numpy as np
import pandas as pd
import torch
import nltk
import transformers as ppb
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Location of SST2 sentiment dataset
SST2_LOC = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'
WEIGHTS = 'distilbert-base-uncased'
# Performance on whole 6920 sentence set is very similar, but takes rather longer
SET_SIZE = 2000

# Download the dataset from its Github location, return as a Pandas dataframe
def get_dataframe():
    df = pd.read_csv(SST2_LOC, delimiter='\t', header=None)
    return df[:SET_SIZE]

# Extract just the labels from the dataframe
def get_labels(df):
    return df[1]

# Get a trained tokenizer for use with BERT
def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)

# Convert the sentences into lists of tokens
def get_tokens(dataframe, tokenizer):
    return dataframe[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# We want the sentences to all be the same length; pad with 0's to make it so
def pad_tokens(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    return padded

# Grab a trained DistiliBERT model
def get_model():
    return ppb.DistilBertModel.from_pretrained(WEIGHTS)

# This step takes a little while, since it actually runs the model on all sentences.
# Get model with get_model(), 0-padded token lists with pad_tokens() on get_tokens().
# Only returns the [CLS] vectors representing the whole sentence, corresponding to first token.
def get_bert_sentence_vectors(model, padded_tokens):
    # Mask the 0's padding from attention - it's meaningless
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens).to(torch.int64), attention_mask=mask)
    # First vector is for [CLS] token, represents the whole sentence
    return word_vecs[0][:,0,:].numpy()


# To separate into train and test:
# train_features, test_features, train_labels, test_labels = train_test_split(vecs, labels)
def train_gaussian_naive_bayes(train_features, train_labels):
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    return gnb

# General purpose scikit-learn classifier evaluator.  The classifier is trained with .fit()
def evaluate(classifier, test_features, test_labels):
    return classifier.score(test_features, test_labels)
    
# Question #1
# takes a list of BERT vectors and a list of corresponding sentences,
# and prints the two (different) sentences that are closest in the space by Euclidean distance.
def find_closest_sentences(vecs,sentences):
    min_dist = np.inf
    size = len(vecs)
    for i in range(size):
        for j in range(i+1,size):
            dist = np.linalg.norm(vecs[i]-vecs[j])
            if dist < min_dist and dist != 0:
                min_dist = dist
                pair = (sentences[i],sentences[j])
    print("The closest sentences are \"{}\" and \"{}\"".format(pair[0],pair[1]))
    print('The minimum distance is {}'.format(min_dist))
    
# Question #3
# performs PCA on the sentences' BERT vector representations
def visualize_data(vecs,labels):
    n = len(vecs)
    pca = PCA(n_components=2)
    data = pca.fit_transform(vecs)
    fig, ax = plt.subplots()
    for i in range(n):
        ax.scatter(data[i][0], data[i][1], c = 'red' if labels[i] == 0 else 'green')
    plt.show()
    
# Question #4
# Train and return an Adaboost learner with decision tree stump weak learners.
def train_adaboost(train_features, train_labels):
    ada = AdaBoostClassifier()
    ada.fit(train_features,train_labels)
    return ada

# "Train" k-nearest neighbors for k=5, and return the classifier.
def train_nearest_neighbors(train_features, train_labels):
    nn = KNeighborsClassifier()
    nn.fit(train_features, train_labels)
    return nn

# Train a classic multilayer neural network ("multilayer perceptron" to scikit-learn)
# with logistic (sigmoid) activation function and 100 hidden units in a single hidden layer.
# Return the classifier
def train_classic_mlp_classifier(train_features, train_labels):
    cmc = MLPClassifier(activation='logistic')
    cmc.fit(train_features, train_labels)
    return cmc

# Not that deep, just 2 hidden layers of 100 units each, using the rectifier activation function.
# Return the classifier.
def train_deep_mlp_classifier(train_features, train_labels):
    dmc = MLPClassifier(hidden_layer_sizes=(100,100), activation='relu')
    dmc.fit(train_features, train_labels)
    return dmc

# Train what's essentially a perceptron with a logistic (sigmoid) activation function; a linear method.
def train_logistic_regression(train_features, train_labels):
    lr = LogisticRegression()
    lr.fit(train_features, train_labels)
    return lr
    
# Question #12
def get_proper_noun(classifier, model, string):
    
    tokenizer0 = get_tokenizer()
    tokens0 = np.array(tokenizer0.encode(string, add_special_tokens=True))
    tokens1 = tokens0.reshape(1,len(tokens0))
    np_array = get_bert_sentence_vectors(model, tokens1)
    sentiment = classifier.predict(np_array)
    
    tokens = nltk.word_tokenize(string)
    tagged = nltk.pos_tag(tokens)
    proper_noun_list = []
    consecutive = []
    i = 0
    while i < len(tokens):
        j = i
        while tagged[j][1] == 'NNP':
            consecutive.append(tagged[j][0])
            j += 1
        if len(consecutive) > 0:
            proper_noun_list.append(consecutive)
            consecutive = []
        i = j + 1
    print("The the proper nouns are \"{}\"".format(proper_noun_list))
    print("The sentiment is {},which is {}".format(sentiment[0],"positive" if sentiment[0] == 1 else "negative"))
    
# main
if __name__ == '__main__':
    df = get_dataframe()
    tokenizer = get_tokenizer()
    tokenized = get_tokens(df,tokenizer)
    pad_tokens = pad_tokens(tokenized)
    model = get_model()
    sentences = df[0]
    vecs = get_bert_sentence_vectors(model, pad_tokens)
    
    
    ##Question #1
    find_closest_sentences(vecs,sentences)
    
    ##Question #3
    visualize_data(vecs,get_labels(df))
    
    ##Question #4
    train_features, test_features, train_labels, test_labels = train_test_split(vecs, get_labels(df))
    print(evaluate(train_gaussian_naive_bayes(train_features,train_labels),test_features,test_labels))
    print(evaluate(train_adaboost(train_features,train_labels),test_features,test_labels))
    print(evaluate(train_nearest_neighbors(train_features,train_labels),test_features,test_labels))
    print(evaluate(train_classic_mlp_classifier(train_features,train_labels),test_features,test_labels))
    print(evaluate(train_deep_mlp_classifier(train_features,train_labels),test_features,test_labels))
    print(evaluate(train_logistic_regression(train_features,train_labels),test_features,test_labels))
    
    ##Question #8
    print(df[0][11])
    
    ##Question #9
    nltk.download('averaged_perceptron_tagger')
    tokens = nltk.word_tokenize(df[0][11])
    tagged = nltk.pos_tag(tokens)
    print(tagged)
    
    ##Question #10
    sentence = "The most repugnant adaptation of a classic text since Roland Joff and Demi Moore 's the scarlet letter"
    tokens_fixed = nltk.word_tokenize(sentence)
    tagged_fixed = nltk.pos_tag(tokens_fixed)
    print(tagged_fixed)
    
    ##Question #11
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    entities = nltk.chunk.ne_chunk(tagged)
    print(entities)
    
    ##Question #12
    ##Test
    get_proper_noun(train_gaussian_naive_bayes(train_features, train_labels),model,sentence)
