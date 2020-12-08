# reference "https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76"
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
np.random.seed(2020)
import random


def get_hashtag(content):
    hashtag = re.findall(r"['\'](.*?)['\']", str(content))
    return hashtag


def readDocument(content_df):
    documents = content_df['content'].tolist()
    for index, doc in enumerate(documents):
        documents[index] = str(doc)
    return documents


def computeTFIDF(documents):
    tf_dict = {}
    vectorizer = TfidfVectorizer(min_df=100, max_df=10000)
    vectors = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    content_df = pd.read_table('./trecData/embedSet.csv')
    documents = content_df['content'].tolist()
    for index, doc in enumerate(documents):
        tf_dict[documents[index]] = df.loc[index].tolist()
    print(tf_dict)
    return tf_dict


def average_user_tweet(user_list, content_user_df, tf_dict):
    user_arr_dict = {}
    for index, user in enumerate(user_list):
        embed_list = []
        content_list = content_user_df['content'].loc[(content_user_df['user_id']) == user].tolist()[0]

        for content in content_list:
            embed_list.append(tf_dict[content])
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list

    print(user_arr_dict)
    print("function: average_user_tweet()")
    return user_arr_dict


def average_hashtag_tweet(tag_list, content_tag_df, tf_dict):
    tag_arr_dict = {}

    for index, tag in enumerate(tag_list):
        print(index)
        embed_list = []
        content_list = content_tag_df['content'].loc[tag == content_tag_df['hashtag']].tolist()[0]

        for content in content_list:
            #print(content)
            embed_list.append(tf_dict[content])
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list

    print(tag_arr_dict)
    print("function: average_hashtag_tweet()")
    return tag_arr_dict


def sort_train_user_tag(user_list, train_df):
    train_df['hashtag'] = train_df['hashtag'].apply(get_hashtag)
    train_tag_list = list(set(train_df['hashtag'].explode('hashtag').tolist()))
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_df = train_df.loc[train_df['user_id'] == user]
        spe_user_tag_list = list(set(spe_user_df['hashtag'].explode('hashtag').tolist()))
        qid_user_tag_dict[user] = spe_user_tag_list

    print(qid_user_tag_dict)
    print("function: sort_train_user_tag()")
    return train_tag_list, qid_user_tag_dict


def sort_test_user_tag(user_list, test_df):
    test_df['hashtag'] = test_df['hashtag'].apply(get_hashtag)
    test_tag_list = list(set(test_df['hashtag'].explode('hashtag').tolist()))
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_df = test_df.loc[test_df['user_id'] == user]
        spe_user_tag_list = list(set(spe_user_df['hashtag'].explode('hashtag').tolist()))
        qid_user_tag_dict[user] = spe_user_tag_list

    print(qid_user_tag_dict)
    print("function: sort_test_user_tag()")
    return test_tag_list, qid_user_tag_dict


def rank_input_train(user_list, train_tag_list, user_arr_dict, tag_arr_dict, qid_train_dict):
    f = open('./trecTf/trainTf.dat', "a")
    for user_num, user in enumerate(user_list):
        print(user_num)
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")
        positive_tag_list = qid_train_dict[user]
        for tag in positive_tag_list: # positive samples
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 1
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)

        temp_tag_list = list(set(train_tag_list)-set(positive_tag_list))
        negative_tag_list = random.sample(temp_tag_list, 5*len(positive_tag_list))
        for tag in negative_tag_list: # negative samples
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 0
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)
        f.write("\n")


def rank_input_test(user_list, test_df, user_arr_dict, tag_arr_dict, qid_test_dict):
    test_df['hashtag'] = test_df['hashtag'].apply(get_hashtag)
    test_df = test_df.explode('hashtag').groupby(['hashtag'], as_index=False)['hashtag'].agg({'cnt': 'count'})
    test_df = test_df.sort_values(by=['cnt'], ascending=False)
    # test_df = test_df[:1000]
    top_tag_list = test_df['hashtag'].tolist()

    f = open('./trecTf/testTf.dat', "a")
    for user_num, user in enumerate(user_list):
        print(user_num)
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")
        positive_tag_list = qid_test_dict[user]
        for tag in positive_tag_list:  # positive samples
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 1
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)

        negative_tag_list = list(set(top_tag_list) - set(positive_tag_list))
        for tag in negative_tag_list:  # negative samples
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = 0
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            f.write(Str)
        f.write("\n")


def read_para(content_df):
    '''
    user_list = list(set(content_df['user_id'].tolist()))
    f = open("userList.txt", "w")
    f.write(str(user_list))
    f.close()
    '''
    with open("userList.txt", "r") as f:
        user_list = get_hashtag(f.readlines()[0])

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
    embedSet = pd.read_table('./trecData/embedSet.csv')
    embedSet['hashtag'] = embedSet['hashtag'].apply(get_hashtag)
    tf_dict = computeTFIDF(readDocument(embedSet))

    user_list, content_user_df, emb_tag_list, content_tag_df = read_para(embedSet)

    user_arr_dict = average_user_tweet(user_list, content_user_df, tf_dict)
    tag_arr_dict = average_hashtag_tweet(emb_tag_list, content_tag_df, tf_dict)

    train_df = pd.read_table('./trecData/trainSet.csv')
    test_df = pd.read_table('./trecData/testSet.csv')
    train_tag_df, qid_train_dict = sort_train_user_tag(user_list, train_df)
    test_tag_df, qid_test_dict = sort_test_user_tag(user_list, test_df)

    rank_input_train(user_list, train_tag_df, user_arr_dict, tag_arr_dict, qid_train_dict)
    rank_input_test(user_list, test_df, user_arr_dict, tag_arr_dict, qid_test_dict)
