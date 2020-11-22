import numpy as np
import pandas as pd
import re
import json
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

    #print(user_arr_dict)
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


def svm_input_train(user_list, tag_list, user_arr_dict, tag_arr_dict, qid_train_dict):
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
            if tag in qid_train_dict[user]:
                x = qid_train_dict[user][tag] #np.random.randint(len(tag_list), size=1).item()
            else:
                x = 0
            f.write(f"\n{x} {'qid'}:{user_num+1}")
            for index, value in enumerate(user_tag_arr):
                f.write(f" {index+1}:{value}")


def svm_input_test(user_list, tag_list, user_arr_dict, tag_arr_dict, qid_test_dict):
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
            if tag in qid_test_dict[user]:
                x = qid_test_dict[user][tag] #np.random.randint(len(tag_list), size=1).item()
            else:
                x = 0
            f.write(f"\n{x} {'qid'}:{user_num+1}")
            for index, value in enumerate(user_tag_arr):
                f.write(f" {index+1}:{value}")


def sort_train_user_tag(user_list, train_df):
    train_df['hashtag'] = train_df['hashtag'].apply(get_hashtag)
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
    return qid_user_tag_dict


def sort_test_user_tag(user_list, test_df):
    test_df['hashtag'] = test_df['hashtag'].apply(get_hashtag)
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
    return qid_user_tag_dict


def read_embedding(embedSet):
    content_df = pd.read_table(embedSet)
    content_df = content_df[:10]
    content_df['hashtag'] = content_df['hashtag'].apply(get_hashtag)
    #print(content_df)
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

    user_arr_dict = average_user_tweet(emb_para_list[0], emb_para_list[1], emb_para_list[4])
    tag_arr_dict = average_hashtag_tweet(emb_para_list[2], emb_para_list[3], emb_para_list[4])

    train_df = pd.read_table('./data/trainSet.csv')
    test_df = pd.read_table('./data/testSet.csv')
    qid_train_dict = sort_train_user_tag(emb_para_list[0], train_df)
    qid_test_dict = sort_test_user_tag(emb_para_list[0], test_df)

    svm_input_train(emb_para_list[0], emb_para_list[2], user_arr_dict, tag_arr_dict, qid_train_dict)
    svm_input_test(emb_para_list[0], emb_para_list[2], user_arr_dict, tag_arr_dict, qid_test_dict)