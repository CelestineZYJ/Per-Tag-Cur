import numpy as np
import pandas as pd
import re
import json
import pdb
from tqdm import tqdm
import random


def get_hashtag(content):
    hashtag = re.findall(r"['\'](.*?)['\']", content)
    return hashtag


def content_embedding(content, con_emb_dict):
    try:
        return con_emb_dict[content]
    except:
        return [0]*768


def average_user_tweet(user_list, content_user_df, con_emb_dict):
    user_arr_dict = {}
    for user in user_list:
        embed_list = []
        content_list = content_user_df['content'].loc[content_user_df['user_id'] == user].tolist()#[0]

        for content in content_list:
            embed_list.append(content_embedding(content, con_emb_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list

    print(user_arr_dict)
    return user_arr_dict


def average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict):
    tag_arr_dict = {}
    for tag in tag_list:
        embed_list = []
        content_list = content_tag_df['content'].loc[content_tag_df['hashtag'] == tag].tolist()#[0]

        for content in content_list:
            embed_list.append(content_embedding(content, con_emb_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list
    #print(tag_arr_dict)
    return tag_arr_dict


def svm_input_train(user_list, tag_list, user_arr_dict, tag_arr_dict):
    f = open('./data/trainSvm.dat', "a")
    for user_num, user in enumerate(user_list):
        fake_dict1 = user_arr_dict[user]
        for index, value in enumerate(fake_dict1):
            fake_dict1[index] = np.random.rand(1, 1).item()
        user_arr = fake_dict1
        #print(user_arr)
        f.write(f"\n# query {user_num+1}")
        for tag in tag_list:
            fake_dict2 = tag_arr_dict[tag]
            #print(fake_dict2)
            for index, value in enumerate(fake_dict2):
                #print(index)
                fake_dict2[index] = random.random()
            tag_arr = fake_dict2
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = np.random.randint(len(tag_list), size=1).item()
            f.write(f"\n{x} {'qid'}:{user_num+1}")
            for index, value in enumerate(user_tag_arr):
                f.write(f" {index+1}:{value}")


def svm_input_test(user_list, tag_list, user_arr_dict, tag_arr_dict):
    f = open('./data/testSvm.dat', "a")
    for user_num, user in enumerate(user_list):
        fake_dict1 = user_arr_dict[user]
        for index, value in enumerate(fake_dict1):
            fake_dict1[index] = np.random.rand(1, 1).item()
        user_arr = fake_dict1
        #print(user_arr)
        for tag in tag_list:
            fake_dict2 = tag_arr_dict[tag]
            #print(fake_dict2)
            for index, value in enumerate(fake_dict2):
                #print(index)
                fake_dict2[index] = random.random()
            tag_arr = fake_dict2
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            x = np.random.randint(len(tag_list), size=1).item()
            f.write(f"\n{x} {'qid'}:{user_num+1}")
            for index, value in enumerate(user_tag_arr):
                f.write(f" {index+1}:{value}")


def sort_train_user_tag(train_df):
    pass


def sort_test_user_tag(test_df):
    pass


def read_embedding(embedSet):
    content_df = pd.read_table(embedSet)
    content_df = content_df[:10]
    content_df['hashtag'] = content_df['hashtag'].apply(get_hashtag)
    print(content_df)
    user_list = list(set(content_df['user_id'].tolist()))
    content_user_df = content_df.groupby(['user_id'], as_index=False).agg({'content': lambda x: list(x)})
    tag_list = list(set(content_df.explode('hashtag')['hashtag'].tolist()))
    content_tag_df = content_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'content': lambda x: list(x)})
    emb_para_list = [user_list, content_user_df, tag_list, content_tag_df]
    return emb_para_list


if __name__ == '__main__':
    with open('./data/embeddings.json', 'r') as f:
        con_emb_dict = json.load(f)
    emb_para_list = read_embedding('./data/embedSet.csv')
    emb_para_list.append(con_emb_dict)

    #user_arr_dict = average_user_tweet(emb_para_list[0], emb_para_list[1], emb_para_list[4])
    #tag_arr_dict = average_hashtag_tweet(emb_para_list[2], emb_para_list[3], emb_para_list[4])

    #svm_input_train(emb_para_list[0], emb_para_list[2], user_arr_dict, tag_arr_dict)
    #svm_input_test(emb_para_list[0], emb_para_list[2], user_arr_dict, tag_arr_dict)

    train_df = pd.read_table('./data/trainSet.csv')
    test_df = pd.read_table('./data/testSet.csv')
    sort_train_user_tag(train_df)
    sort_test_user_tag(test_df)