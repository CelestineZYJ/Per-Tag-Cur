# reference "https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76"
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def readDocument(content_df):
    documents = content_df['content'][:2000].tolist()
    return documents


def computeTFIDF(documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    print(df)
    df.to_csv('./data/tfidf.csv')


if __name__ == '__main__':
    content_df = pd.read_table('./data/testSet.csv')
    computeTFIDF(readDocument(content_df))



'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math


def computeTFIDF(document, idfDict):
    tfidf = {}
    for word, val in document.items():
        tfidf[word] = val*idfDict[word]
    print(tfidf)


def computeIDF(documents):
    N = len(documents)

    idfDict = dict.fromkeys(documents[0], 0)
    for document in documents:
        for word, val in document.items():
            if word in idfDict:
                idfDict[word] += 1
            else:
                idfDict[word] = 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N/float(val))
    return idfDict


if __name__ == "__main__":
    document1 = 'the man went out for a walk'
    document2 = 'the children sat around the fire'
    documents = [document1, document2]
    for i in range(len(documents)):
        documents[i] = dict.fromkeys(set(documents[i].split(' ')), 1)

    for document in documents:
        computeTFIDF(document, computeIDF(documents))
'''