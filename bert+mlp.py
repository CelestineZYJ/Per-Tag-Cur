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


def average_user_tweet(user_list, content_user_df, con_emb_dict):
    user_arr_dict = {}
    for user in user_list:
        embed_list = []
        x = content_user_df['content'].loc[(content_user_df['user_id']) == user].tolist()
        #print(x)
        content_list = x[0]

        for content in content_list:
            embed_list.append(content_embedding(content, con_emb_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list

    print(user_arr_dict)
    print("function: average_user_tweet()")
    return user_arr_dict


def average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict):
    tag_arr_dict = {}
    #print(len(tag_list))
    for index, tag in enumerate(tag_list):
        #print(str(index)+tag)
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


def sort_valid_user_tag(user_list, valid_df):
    valid_df['hashtag'] = valid_df['hashtag'].apply(get_hashtag)
    valid_tag_list = list(set(valid_df['hashtag'].explode('hashtag').tolist()))
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_df = valid_df.loc[valid_df['user_id'] == user]
        spe_user_tag_list = list(set(spe_user_df['hashtag'].explode('hashtag').tolist()))
        qid_user_tag_dict[user] = spe_user_tag_list

    print(qid_user_tag_dict)
    return valid_tag_list, qid_user_tag_dict


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
    with open("tData2/userList.txt", "r") as f:
        x = f.readlines()[0]
        #print(x)
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


train_df = pd.read_table('./tData2/train.csv')
test_df = pd.read_table('./tData2/test.csv')
valid_df = pd.read_table('./tData2/validation.csv')

# 这几个get_str是为了应对中文数据集经常读出来非str的问题，跑trec的时候注释掉这几句，不然会报错，原因待调查

train_df['user_id'] = train_df['user_id'].apply(get_str)
test_df['user_id'] = test_df['user_id'].apply(get_str)
valid_df['user_id'] = valid_df['user_id'].apply(get_str)
train_df['content'] = train_df['content'].apply(get_str)
test_df['content'] = test_df['content'].apply(get_str)
valid_df['content'] = valid_df['content'].apply(get_str)

with open('./tData2/embeddings.json', 'r') as f:
    con_emb_dict = json.load(f)

embedSet = pd.read_table('./tData2/embed.csv')
# 这几个get_str是为了应对中文数据集经常读出来非str的问题，跑trec的时候注释掉这几句，不然会报错，原因待调查
embedSet['user_id'] = embedSet['user_id'].apply(get_str)
embedSet['content'] = embedSet['content'].apply(get_str)
embedSet['hashtag'] = embedSet['hashtag'].apply(get_hashtag)
user_list, content_user_df, tag_list, content_tag_df = read_embedding(embedSet, test_df)

user_arr_dict = average_user_tweet(user_list, content_user_df, con_emb_dict)
tag_arr_dict = average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict)

train_tag_list, qid_train_dict = sort_train_user_tag(user_list, train_df)
valid_tag_list, qid_valid_dict = sort_valid_user_tag(user_list, valid_df)
test_tag_list, qid_test_dict = sort_test_user_tag(user_list, test_df)


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size*2, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


def all_user(user_list):
    user_num = 0
    for user_id in tqdm(user_list[:149]):
        user_arr = user_arr_dict[user_id]  # type nparr

        # train
        feature_train = []
        label_train = []
        # positive samples
        positive_tag_list = qid_train_dict[user_id]
        for tag in positive_tag_list:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            feature_train.append(user_tag_arr)
            x = 1
            label_train.append(x)  # positive sample label: 1

        # negative samples
        temp_tag_list = list(set(train_tag_list) - set(positive_tag_list))
        negative_tag_list = random.sample(temp_tag_list, 5 * len(positive_tag_list))
        for tag in negative_tag_list:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            feature_train.append(user_tag_arr)
            x = 0
            label_train.append(x)  # negative sample label: 0

        feature_train = np.array(feature_train)
        label_train = np.array(label_train)

        feature_train = torch.FloatTensor(feature_train)
        label_train = torch.FloatTensor(label_train)

        # validate
        feature_valid = []
        label_valid = []
        # positive samples
        positive_tag_list = qid_valid_dict[user_id]
        for tag in positive_tag_list:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            feature_valid.append(user_tag_arr)
            x = 1
            label_valid.append(x)  # positive sample label: 1

        # negative samples
        temp_tag_list = list(set(valid_tag_list) - set(positive_tag_list))
        negative_tag_list = random.sample(temp_tag_list, 5 * len(positive_tag_list))
        for tag in negative_tag_list:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            feature_valid.append(user_tag_arr)
            x = 0
            label_valid.append(x)  # negative sample label: 0

        feature_valid = np.array(feature_valid)
        label_valid = np.array(label_valid)

        feature_valid = torch.FloatTensor(feature_valid)
        label_valid = torch.FloatTensor(label_valid)

        # test
        testF = open('./tBertMlp/testBertMlp.dat', "a")
        testF.write(f"# query {user_num + 1}")

        testF2 = open('./tBertMlp/testBertMlp2.dat', "a")
        testF2.write(f"# query {user_num + 1}")

        feature_test = []
        label_test = []
        # positive samples
        positive_tag_list = qid_train_dict[user_id]
        for tag in positive_tag_list:
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            feature_test.append(user_tag_arr)
            x = 1
            label_test.append(x)  # positive sample label: 1

            # write qid test file
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            testF2.write(Str)
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            testF.write(Str)

        # negative samples
        negative_tag_list = list(set(test_tag_list) - set(positive_tag_list))
        for tag in negative_tag_list:  # negative samples
            tag_arr = tag_arr_dict[tag]
            user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
            feature_test.append(user_tag_arr)
            x = 0
            label_test.append(x)  # negative sample label: 0

            # write qid test file
            Str = f"\n{x} {'qid'}:{user_num + 1}"
            testF2.write(Str)
            for index, value in enumerate(user_tag_arr):
                Str += f" {index + 1}:{value}"
            testF.write(Str)
        testF.write("\n")
        testF2.write("\n")
        testF.close()
        testF2.close()

        # print(feature_test[0])
        # print(type(feature_test[0]))
        # print(len(feature_test[0]))

        feature_test = np.array(feature_test)
        label_test = np.array(label_test)

        feature_test = torch.FloatTensor(feature_test)
        label_test = torch.FloatTensor(label_test)

        each_user(feature_train, label_train, feature_valid, label_valid, feature_test, label_test)

        user_num += 1


def each_user(feature_train, label_train, feature_valid, label_valid, feature_test, label_test):
    # model, criterion, optimizer
    model = Feedforward(768, 30)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, verbose=False)

    # evaluate before train
    model.eval()
    label_pred = model(feature_test)
    before_train = criterion(label_pred.squeeze(), label_test)
    print("\ntest loss before training", before_train.item())

    # train the model
    model.train()
    epoch = 50

    for epoch in range(epoch):
        # train process-----------------------------------
        optimizer.zero_grad()

        # forward pass
        label_pred = model(feature_train)

        # compute loss
        loss = criterion(label_pred.squeeze(), label_train)

        # backward pass
        loss.backward()
        optimizer.step()

        # validate process----------------------------------
        optimizer.zero_grad()
        label_pred = model(feature_valid)
        val_loss = criterion(label_pred.squeeze(), label_valid)
        scheduler.step(val_loss)

    # evalution
    model.eval()
    label_pred = model(feature_test)

    preF = open('tBertMlp/preBertMlp.txt', "a")
    spe_user_pre = label_pred.detach().numpy().tolist()
    for tag_pre in spe_user_pre:
        preF.write(f"{tag_pre[0]}\n")
    preF.close()

    print(label_pred.squeeze())
    print(label_test)
    after_train = criterion(label_pred.squeeze(), label_test)
    print("test loss after training", after_train.item())


all_user(user_list)