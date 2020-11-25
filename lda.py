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
    hashtag = re.findall(r"['\'](.*?)['\']", str(content))
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
        doc_list = []
        for index, score in sorted(lda_model[Value], key=lambda tup: -1 * tup[1]):
            #print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
            doc_list.append(score)
        if len(doc_list) < 10:
            for i in range(10-len(doc_list)):
                doc_list.append(float(0))
        lda_dict[sentences[Index]] = doc_list

    #print(lda_dict)
    return lda_dict


def content_lda(content, lda_dict):
    return lda_dict[content]


def average_user_tweet(user_list, content_user_df, lda_dict):
    user_arr_dict = {}
    for index, user in enumerate(user_list):
        embed_list = []
        content_list = content_user_df['content'].loc[(content_user_df['user_id']) == user].tolist()[0]

        for content in content_list:
            embed_list.append(content_lda(content, lda_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list

    print(user_arr_dict)
    return user_arr_dict


def average_hashtag_tweet(tag_list, content_tag_df, lda_dict):
    tag_arr_dict = {}

    for index, tag in enumerate(tag_list):
        embed_list = []
        content_list = content_tag_df['content'].loc[tag == content_tag_df['hashtag']].tolist()[0]

        for content in content_list:
            #print(content)
            embed_list.append(content_lda(content, lda_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list

    print(tag_arr_dict)
    return tag_arr_dict


def sort_train_user_tag(user_list, train_df):
    train_df['hashtag'] = train_df['hashtag'].apply(get_hashtag)
    train_tag_list = list(set(train_df['hashtag'].explode('hashtag').tolist()))
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_dict = {}
        spe_user_df = train_df.loc[train_df['user_id'] == user]
        spe_user_df = spe_user_df.sort_values(by=['time'], ascending=True)
        spe_user_tag_list = spe_user_df['hashtag'].tolist()
        for index, value in enumerate(spe_user_tag_list):
            for tag in value:
                spe_user_dict[tag] = index + 1
        qid_user_tag_dict[user] = spe_user_dict

    print(qid_user_tag_dict)
    return train_tag_list, qid_user_tag_dict


def sort_test_user_tag(user_list, test_df):
    test_df['hashtag'] = test_df['hashtag'].apply(get_hashtag)
    test_tag_list = list(set(test_df['hashtag'].explode('hashtag').tolist()))
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_dict = {}
        spe_user_df = test_df.loc[test_df['user_id'] == user]
        spe_user_df = spe_user_df.sort_values(by=['time'], ascending=True)
        spe_user_tag_list = spe_user_df['hashtag'].tolist()
        for index, value in enumerate(spe_user_tag_list):
            for tag in value:
                spe_user_dict[tag] = index+1
        qid_user_tag_dict[user] = spe_user_dict

    print(qid_user_tag_dict)
    return test_tag_list, qid_user_tag_dict


def rank_input_train(user_list, train_tag_list, user_arr_dict, tag_arr_dict, qid_train_dict):
    f = open('./data/trainSvm.dat', "a")
    for user_num, user in enumerate(user_list):
        user_arr = user_arr_dict[user]
        #print(user_arr)
        f.write(f"\n# query {user_num+1}")
        for tag in train_tag_list:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            if tag in qid_train_dict[user]:
                x = qid_train_dict[user][tag]
            else:
                x = 0
            f.write(f"\n{x} {'qid'}:{user_num+1}")
            for index, value in enumerate(user_tag_arr):
                f.write(f" {index+1}:{value}")


def rank_input_test(user_list, test_tag_list, user_arr_dict, tag_arr_dict, qid_test_dict):
    f = open('./data/testSvm.dat', "a")
    for user_num, user in enumerate(user_list):
        user_arr = user_arr_dict[user]
        # print(user_arr)
        for tag_num, tag in enumerate(test_tag_list):
            print('user_num: '+str(user_num)+'  tag_num: '+str(tag_num))
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            if tag in qid_test_dict[user]:
                x = qid_test_dict[user][tag]
            else:
                x = 0
            # f.write(f"\n{x} {'qid'}:{user_num+1}")
            Str = f"\n{x} {'qid'}:{user_num+1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index+1}:{value}"
        f.write(Str)


def read_para(content_df):
    # print(content_df)
    user_list = list(set(content_df['user_id'].tolist()))
    content_user_df = content_df.groupby(['user_id'], as_index=False).agg({'content': lambda x: list(x)})
    '''
    tag_list = list(set(content_df.explode('hashtag')['hashtag'].tolist()))
    temp = content_df.explode('hashtag')
    temp['hashtag'] = temp['hashtag'].apply(get_hashtag)
    # print(temp)
    for index, tag in enumerate(temp):
        if str(tag) == 'nan':
            print(index)
    '''
    content_tag_df = content_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'content': lambda x: list(x)})
    tag_list = list(set(content_tag_df['hashtag'].tolist()))

    return user_list, content_user_df, tag_list, content_tag_df


if __name__ == '__main__':
    embedSet = pd.read_table('./data/embedSet.csv')
    #embedSet = embedSet[:1000]
    embedSet['hashtag'] = embedSet['hashtag'].apply(get_hashtag)
    user_list, content_user_df, emb_tag_list, content_tag_df = read_para(embedSet)
    lda_dict = lda(embedSet)

    user_arr_dict = average_user_tweet(user_list, content_user_df, lda_dict)
    tag_arr_dict = average_hashtag_tweet(emb_tag_list, content_tag_df, lda_dict)

    train_df = pd.read_table('./data/trainSet.csv')
    test_df = pd.read_table('./data/testSet.csv')
    train_tag_df, qid_train_dict = sort_train_user_tag(user_list, train_df)
    test_tag_df, qid_test_dict = sort_test_user_tag(user_list, test_df)

    rank_input_train(user_list, train_tag_df, user_arr_dict, tag_arr_dict, qid_train_dict)
    rank_input_test(user_list, test_tag_df, user_arr_dict, tag_arr_dict, qid_test_dict)