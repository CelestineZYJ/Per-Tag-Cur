# reference "https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24"
import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2020)
import nltk
# nltk.download('wordnet')


def get_hashtag(content):
    hashtag = re.findall(r"['\'](.*?)['\']", content)
    return hashtag


def readDocument(content_df):
    documents = content_df['content'].tolist()
    return documents


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
    # print(documents[1])
    return documents


def lda(content_df):
    lda_dict = {}
    sentences = readDocument(content_df)
    documents = whole_preproess(content_df)
    dictionary = gensim.corpora.Dictionary(documents)

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2) # len(documents) == len(bow_corpus)

    # for idx, topic in lda_model.print_topics(-1):
        # print('Topic: {} \nWords: {}'.format(idx, topic))

    for Index, Value in enumerate(bow_corpus):
        #print(Index)
        doc_list = []
        for index, score in sorted(lda_model[Value], key=lambda tup: -1 * tup[1]):
            #print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
            doc_list.append((score))
        #print(doc_list)
        lda_dict[sentences[Index]] = doc_list

    #print(lda_dict)
    return lda_dict


def content_lda(content, lda_dict):
    return lda_dict[content]


def average_user_tweet(user_list, content_user_df, lda_dict):
    user_arr_dict = {}
    for user in user_list:
        embed_list = []
        content_list = content_user_df['content'].loc[content_user_df['user_id'] == user].tolist()[0]
        print(content_list)
        print('okkkkkkkkkkkkkkkkk')

        for content in content_list:
            print(content)
            print('aaa')
            embed_list.append(content_lda(content, lda_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list

    print(user_arr_dict)
    return user_arr_dict


def average_hashtag_tweet(tag_list, content_tag_df, lda_dict):
    tag_arr_dict = {}
    for tag in tag_list:
        embed_list = []
        content_list = content_tag_df['content'].loc[content_tag_df['hashtag'] == tag].tolist()[0]

        for content in content_list:
            embed_list.append(content_lda(content, lda_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list

    print(tag_arr_dict)
    return tag_arr_dict


def read_para(content_df):
    # print(content_df)
    user_list = list(set(content_df['user_id'].tolist()))
    content_user_df = content_df.groupby(['user_id'], as_index=False).agg({'content': lambda x: list(x)})
    tag_list = list(set(content_df.explode('hashtag')['hashtag'].tolist()))
    content_tag_df = content_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'content': lambda x: list(x)})

    return user_list, content_user_df, tag_list, content_tag_df


if __name__ == '__main__':
    embedSet = pd.read_table('./data/embedSet.csv')
    embedSet = embedSet[:1000]
    embedSet['hashtag'] = embedSet['hashtag'].apply(get_hashtag)
    user_list, content_user_df, emb_tag_list, content_tag_df = read_para(embedSet)
    lda_dict = lda(embedSet)

    average_user_tweet(user_list, content_user_df, lda_dict)
    #average_hashtag_tweet(emb_tag_list, content_tag_df, lda_dict)