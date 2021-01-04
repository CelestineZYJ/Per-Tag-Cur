#  -*- coding: utf-8 -*-
import pandas as pd
import re
import json
import random
import numpy as np
import torch
import re
from tqdm import tqdm


def get_hashtag(content):
    hashtag = re.findall(r"['\'](.*?)['\']", str(content))
    return hashtag


def get_user(content):
    user = re.split(r"[\[\],]", str(content))
    return user[1:-1]


def get_str(content):
    Str = str(content)
    return Str


def content_embedding(content, con_emb_dict):
    #try:
        return con_emb_dict[content]
    #except:
        #return [0]*768


def average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict):
    tag_arr_dict = {}
    print(len(tag_list))
    for index, tag in enumerate(tag_list):
        print(str(index)+tag)
        embed_list = []
        content_list = content_tag_df['content'].loc[(content_tag_df['hashtag']) == tag].tolist()[0]

        for content in content_list:
            embed_list.append(content_embedding(content, con_emb_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list
        #print(tag_arr_dict[tag])

    #print(tag_arr_dict)
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
    return test_tag_list, qid_user_tag_dict


def read_embedding(content_df, test_df):
    # 写userList
    '''
    user_list = list(set(test_df['user_id'].tolist()))
    f = open("wData/userList.txt", "w")
    f.write(str(user_list))
    f.close()

    # 读userlist，要灵活调换写与读以保持与其他实验的统一
    '''
    with open("tData/userList.txt", "r") as f:
        x = f.readlines()[0]
        print(x)
        user_list = get_hashtag(x)
        print(user_list)

    content_user_df = content_df.groupby(['user_id'], as_index=False).agg({'content': lambda x: list(x)})
    content_tag_df = content_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'content': lambda x: list(x)})
    tag_list = list(set(content_tag_df['hashtag'].tolist()))

    '''
    train_df = pd.read_table('./data/trainSet.csv')
    train_df['hashtag'] = train_df['hashtag'].apply(get_hashtag)
    train_tag_list = list(set(train_df['hashtag'].explode('hashtag').tolist()))
    print(train_tag_list)
    print(tag_list)


    for tag in train_tag_list:
        if tag not in tag_list:
            print(tag)
    '''
    print("user_num: " + str(len(user_list)))
    print("tag_num: " + str(len(tag_list)))
    return user_list, content_user_df, tag_list, content_tag_df


train_df = pd.read_table('./tData/train.csv')
test_df = pd.read_table('./tData/test.csv')

# 这几个get_str是为了应对中文数据集经常读出来非str的问题，跑trec的时候注释掉这几句，不然会报错，原因待调查

train_df['user_id'] = train_df['user_id'].apply(get_str)
test_df['user_id'] = test_df['user_id'].apply(get_str)
train_df['content'] = train_df['content'].apply(get_str)
test_df['content'] = test_df['content'].apply(get_str)

with open('./tData/embeddings.json', 'r') as f:
    con_emb_dict = json.load(f)

embedSet = pd.read_table('./tData/embed.csv')
# 这几个get_str是为了应对中文数据集经常读出来非str的问题，跑trec的时候注释掉这几句，不然会报错，原因待调查
embedSet['user_id'] = embedSet['user_id'].apply(get_str)
embedSet['content'] = embedSet['content'].apply(get_str)
embedSet['hashtag'] = embedSet['hashtag'].apply(get_hashtag)
user_list, content_user_df, tag_list, content_tag_df = read_embedding(embedSet, test_df)

tag_arr_dict = average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict)

train_tag_list, qid_train_dict = sort_train_user_tag(user_list, train_df)
test_tag_list, qid_test_dict = sort_test_user_tag(user_list, test_df)


class LstmMlp(torch.nn.Module):
    def __init__(self, input_size, hidden_size, user_id):
        super(LstmMlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.user_id = user_id
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # user modeling: calculate user embedding
        lstm = torch.nn.LSTM(self.input_size, self.input_size)
        lstm_inputs = []
        content_list = content_user_df['content'].loc[(content_user_df['user_id']) == self.user_id].tolist()[0]
        for content in content_list:
            lstm_inputs.append(torch.tensor([content_embedding(content, con_emb_dict)]))

        lstm_inputs = torch.cat(lstm_inputs).view(len(lstm_inputs), 1, -1)
        lstm_output = lstm(lstm_inputs)
        user_arr = lstm_output.numpy()  # lstm_output is user modeling

        if x == "train":
            feature_train = []
            label_train = []
            # positive samples
            positive_tag_list = qid_train_dict[self.user_id]
            for tag in positive_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_train.append(user_tag_arr)
                label_train.append(1)  # positive sample label: 1

            # negative samples
            temp_tag_list = list(set(train_tag_list)-set(positive_tag_list))
            negative_tag_list = random.sample(temp_tag_list, 5*len(positive_tag_list))
            for tag in negative_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_train.append(user_tag_arr)
                label_train.append(0)  # negative sample label: 0
            mlp_input = torch.tensor(feature_train)
            mlp_label = torch.tensor(label_train)

        if x == "test":
            feature_test = []
            label_test = []
            # positive samples
            positive_tag_list = qid_train_dict[self.user_id]
            for tag in positive_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_train.append(user_tag_arr)
                label_train.append(1)  # positive sample label: 1

            # negative samples
            negative_tag_list = list(set(test_tag_list) - set(positive_tag_list))
            for tag in negative_tag_list:  # negative samples
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_train.append(user_tag_arr)
                label_train.append(0)  # negative sample label: 0
            mlp_input = torch.tensor(feature_test)
            mlp_label = torch.tensor(label_test)

        mlp_hidden = self.fc1(mlp_input)
        mlp_relu = self.relu(mlp_hidden)
        mlp_output = self.fc2(mlp_relu)
        mlp_output = self.sigmoid(mlp_output)
        return mlp_output, mlp_label