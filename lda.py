# reference "https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24"
from typing import List, Any

import pandas as pd
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2020)
import random
#import nltk
#nltk.download('wordnet')


def get_hashtag(content):
    hashtag = re.findall(r"['\'](.*?)['\']", str(content))
    return hashtag


def get_str(content):
    Str = str(content)
    return Str


def get_user(content):
    user = re.split(r"[\[\],]", str(content))
    return user[1:-1]


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
        try:
            documents[i] = sin_preprocess(str(documents[i]))
        except:
            print(str(i)+": "+str(documents[i]))
            print(content_df[i-1:i]['content'])
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
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=50, id2word=dictionary, passes=2, workers=2) # len(documents) == len(bow_corpus)

    # for idx, topic in lda_model.print_topics(-1):
        # print('Topic: {} \nWords: {}'.format(idx, topic))

    for Index, Value in enumerate(bow_corpus):
        doc_list = []
        for index, score in sorted(lda_model[Value], key=lambda tup: -1 * tup[1]):
            #print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
            doc_list.append(score)
        if len(doc_list) < 50:
            for i in range(50-len(doc_list)):
                doc_list.append(float(0))
        lda_dict[sentences[Index]] = doc_list

    #print(lda_dict)
    print("function: lda()")
    return lda_dict


def content_lda(content, lda_dict):
    return lda_dict[content]


def average_user_tweet(user_list, content_user_df, lda_dict):
    user_arr_dict = {}
    for index, user in enumerate(user_list):
        embed_list = []
        x = content_user_df['content'].loc[(content_user_df['user_id']) == user].tolist()
        print(x)
        content_list = x[0]

        for content in content_list:
            embed_list.append(content_lda(content, lda_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list

    #print(user_arr_dict)
    print("function: average_user_tweet()")
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

    #print(tag_arr_dict)
    print("function: average_hashtag_tweet()")
    return tag_arr_dict


def sort_train_user_tag(user_list, train_df):
    train_df['hashtag'] = train_df['hashtag'].apply(get_hashtag)
    train_tag_list = list(set(train_df['hashtag'].explode('hashtag').tolist()))
    '''
    for index, tag in enumerate(train_tag_list):
        if str(tag) == "nan":
            print(str(index)+" "+str(tag) + " train_tag_list\n")
    '''
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_df = train_df.loc[train_df['user_id'] == user]
        spe_user_tag_list = list(set(spe_user_df['hashtag'].explode('hashtag').tolist()))
        for tag in spe_user_tag_list:
            if str(tag) == "nan":
                print(user)
                print(tag)
        qid_user_tag_dict[user] = spe_user_tag_list

    #print(qid_user_tag_dict)
    print("function: sort_train_user_tag()")
    return train_tag_list[1:], qid_user_tag_dict


def sort_test_user_tag(user_list, test_df):
    test_df['hashtag'] = test_df['hashtag'].apply(get_hashtag)
    test_tag_list = list(set(test_df['hashtag'].explode('hashtag').tolist()))
    '''
    for index, tag in enumerate(test_tag_list):
        if str(tag) == "nan":
            print(str(index)+" "+str(tag) + " test_tag_list\n")
    '''
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_df = test_df.loc[test_df['user_id'] == user]
        spe_user_tag_list = list(set(spe_user_df['hashtag'].explode('hashtag').tolist()))
        for tag in spe_user_tag_list:
            if str(tag) == "nan":
                print(user)
                print(tag)
        qid_user_tag_dict[user] = spe_user_tag_list

    #print(qid_user_tag_dict)
    print("function: sort_test_user_tag()")
    return test_tag_list[1:], qid_user_tag_dict


def rank_input_train(user_list, train_tag_list, user_arr_dict, tag_arr_dict, qid_train_dict):
    f = open('./wLda/trainLda.dat', "a")
    for user_num, user in enumerate(user_list):
        print(user_num)
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")
        positive_tag_list = qid_train_dict[user]
        for tag in positive_tag_list:  # positive samples
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 1
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)

        temp_tag_list = list(set(train_tag_list)-set(positive_tag_list))
        negative_tag_list = random.sample(temp_tag_list, 5*len(positive_tag_list))

        for tag in negative_tag_list:  # negative samples
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 0
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)
        f.write("\n")


def rank_input_test(user_list, test_tag_list, user_arr_dict, tag_arr_dict, qid_test_dict):
    '''
    test_df['hashtag'] = test_df['hashtag'].apply(get_hashtag)
    test_df = test_df.explode('hashtag').groupby(['hashtag'], as_index=False)['hashtag'].agg({'cnt': 'count'})
    test_df = test_df.sort_values(by=['cnt'], ascending=False)
    # test_df = test_df[:1000]
    top_tag_list = test_df['hashtag'].tolist()
    '''
    f = open('./wLda/testLda.dat', "a")
    for user_num, user in enumerate(user_list):
        print(user_num)
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")
        positive_tag_list = qid_test_dict[user]
        for tag in positive_tag_list:  # positive samples
            try:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                x = 1
                Str = f"\n{x} {'qid'}:{user_num + 1}"
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                f.write(Str)
            except:
                print(tag)

        negative_tag_list = list(set(test_tag_list) - set(positive_tag_list))
        for tag in negative_tag_list:  # negative samples
            try:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                x = 0
                Str = f"\n{x} {'qid'}:{user_num + 1}"
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                f.write(Str)
            except:
                print(tag)
        f.write("\n")


def read_para(content_df):
    '''
    user_list = list(set(content_df['user_id'].tolist()))
    f = open("wData/userList.txt", "w")
    f.write(str(user_list))
    f.close()
    '''
    with open("wData/userList.txt", "r") as f:
        x = f.readlines()[0]
        print(x)
        user_list = get_hashtag(x)
        print(user_list)

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
    print("user_num: "+str(len(user_list)))
    print("tag_num: " + str(len(tag_list)))
    return user_list, content_user_df, tag_list, content_tag_df


if __name__ == '__main__':
    embedSet = pd.read_table('./wData/embed.csv')
    #embedSet = embedSet[:10000]
    embedSet['hashtag'] = embedSet['hashtag'].apply(get_hashtag)
    embedSet['user_id'] = embedSet['user_id'].apply(get_str)
    embedSet['content'] = embedSet['content'].apply(get_str)
    user_list, content_user_df, emb_tag_list, content_tag_df = read_para(embedSet)
    lda_dict = lda(embedSet)

    user_arr_dict = average_user_tweet(user_list, content_user_df, lda_dict)
    tag_arr_dict = average_hashtag_tweet(emb_tag_list, content_tag_df, lda_dict)

    train_df = pd.read_table('./wData/train.csv')
    test_df = pd.read_table('./wData/test.csv')
    train_df['user_id'] = train_df['user_id'].apply(get_str)
    test_df['user_id'] = test_df['user_id'].apply(get_str)
    train_df['content'] = train_df['content'].apply(get_str)
    test_df['content'] = test_df['content'].apply(get_str)
    train_tag_df, qid_train_dict = sort_train_user_tag(user_list, train_df)
    test_tag_df, qid_test_dict = sort_test_user_tag(user_list, test_df)

    rank_input_train(user_list, train_tag_df, user_arr_dict, tag_arr_dict, qid_train_dict)
    rank_input_test(user_list, test_tag_df, user_arr_dict, tag_arr_dict, qid_test_dict)
