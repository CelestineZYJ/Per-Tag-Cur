# reference "https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24"
import pandas as pd
import gensim
# from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2020)
import nltk
# nltk.download('wordnet')


def readDocument(content_df):
    documents = content_df['content'].tolist()
    return documents[:1000]


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def sin_preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


def whole_preproess(content_df):
    documents = readDocument(content_df)
    for i in range(len(documents)):
        documents[i] = sin_preprocess(documents[i])
        # print(documents[i])
    print(type(documents[1]))
    return documents


def lda(content_df):
    documents = whole_preproess(content_df)
    dictionary = gensim.corpora.Dictionary(documents)
    ''' 
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
    '''
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100)
    # print(dictionary)
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    print(lda_model)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))


if __name__ == '__main__':
    content_df = pd.read_table('./data/testSet.csv')
    lda(content_df)