#  -*- coding: utf-8 -*-
import pandas as pd
import re
import json
import random
import numpy as np
import torch
import re
from tqdm import tqdm


modelPath = 'demoLstm'
dataPath = 'demo'


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


def sort_embed_user_tag(user_list, embed_df):
    embed_df['hashtag'] = embed_df['hashtag'].apply(get_hashtag)
    embed_tag_list = list(set(embed_df['hashtag'].explode('hashtag').tolist()))
    qid_user_tag_dict = {}
    for user in user_list:
        spe_user_df = embed_df.loc[embed_df['user_id'] == user]
        spe_user_tag_list = list(set(spe_user_df['hashtag'].explode('hashtag').tolist()))
        qid_user_tag_dict[user] = spe_user_tag_list

    print(qid_user_tag_dict)
    return embed_tag_list, qid_user_tag_dict


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
    with open(dataPath+"/userList.txt", "r") as f:
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


train_df = pd.read_table('./'+dataPath+'/train.csv')
test_df = pd.read_table('./'+dataPath+'/test.csv')
valid_df = pd.read_table('./'+dataPath+'/validation.csv')

# 这几个get_str是为了应对中文数据集经常读出来非str的问题，跑trec的时候注释掉这几句，不然会报错，原因待调查

train_df['user_id'] = train_df['user_id'].apply(get_str)
test_df['user_id'] = test_df['user_id'].apply(get_str)
valid_df['user_id'] = valid_df['user_id'].apply(get_str)
train_df['content'] = train_df['content'].apply(get_str)
test_df['content'] = test_df['content'].apply(get_str)
valid_df['content'] = valid_df['content'].apply(get_str)

with open('./'+dataPath+'/embeddings.json', 'r') as f:
    con_emb_dict = json.load(f)

embedSet = pd.read_table('./'+dataPath+'/embed.csv')
# 这几个get_str是为了应对中文数据集经常读出来非str的问题，跑trec的时候注释掉这几句，不然会报错，原因待调查
embedSet['user_id'] = embedSet['user_id'].apply(get_str)
embedSet['content'] = embedSet['content'].apply(get_str)
embedSet['hashtag'] = embedSet['hashtag'].apply(get_hashtag)
embedSet['time'] = embedSet['time'].apply(get_str)
user_list, content_user_df, tag_list, content_tag_df = read_embedding(embedSet, test_df)

tag_arr_dict = average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict)

embed_tag_list, qid_embed_dict = sort_embed_user_tag(user_list, embedSet[embedSet.time < '20200601'])  # trec 20110201
train_tag_list, qid_train_dict = sort_train_user_tag(user_list, train_df)
valid_tag_list, qid_valid_dict = sort_valid_user_tag(user_list, valid_df)
test_tag_list, qid_test_dict = sort_test_user_tag(user_list, test_df)


class LstmMlp(torch.nn.Module):
    def __init__(self, user_num, user_id, input_size, hidden_size):
        super(LstmMlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.user_num = user_num
        self.user_id = user_id
        self.fc1 = torch.nn.Linear(self.input_size*2, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # user modeling: calculate user embedding
        lstm = torch.nn.LSTM(self.input_size, self.input_size)
        lstm_inputs = []
        content_list = content_user_df['content'].loc[(content_user_df['user_id']) == self.user_id].tolist()[0]
        for content in content_list:
            lstm_inputs.append(torch.tensor(content_embedding(content, con_emb_dict)))

        lstm_output, lstm_hidden = lstm(lstm_inputs[0].view(1, 1, -1))
        for i in lstm_inputs[:-1]:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            lstm_output, lstm_hidden = lstm(i.view(1, 1, -1), lstm_hidden)

        lstm_output, lstm_hidden = lstm(lstm_inputs[-1].view(1, 1, -1), lstm_hidden)

        user_arr = lstm_output.detach().numpy()[0][0]  # lstm_output is user modeling
        #print(user_arr)
        #print(len(user_arr)) #768

        if x == "train":
            '''
            trainF = open('./tBert/trainBertLstm.dat', "a")
            trainF.write(f"# query {self.user_num + 1}")
            '''
            feature_train = []
            label_train = []
            # positive samples
            positive_tag_list = sorted(list(set(qid_train_dict[self.user_id])-set(qid_embed_dict[self.user_id])))
            for tag in positive_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_train.append(user_tag_arr)
                x = 1
                label_train.append(x)  # positive sample label: 1
                '''
                # write qid train file
                Str = f"\n{x} {'qid'}:{self.user_num + 1}"
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                trainF.write(Str)
                '''
            # negative samples
            temp_tag_list = list(set(train_tag_list) - set(positive_tag_list)-set(qid_embed_dict[self.user_id]))
            negative_tag_list = random.sample(temp_tag_list, 5*len(positive_tag_list))
            for tag in negative_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_train.append(user_tag_arr)
                x = 0
                label_train.append(x)  # negative sample label: 0
            '''
                # write qid train file
                Str = f"\n{x} {'qid'}:{self.user_num + 1}"
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                trainF.write(Str)
            trainF.write("\n")
            trainF.close()
            '''
            feature_train = np.array(feature_train)
            label_train = np.array(label_train)

            mlp_input = torch.FloatTensor(feature_train)
            mlp_label = torch.FloatTensor(label_train)

        if x == "validation":
            feature_valid = []
            label_valid = []
            # positive samples
            positive_tag_list = sorted(list(set(qid_valid_dict[self.user_id])-set(qid_embed_dict[self.user_id])))
            for tag in positive_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_valid.append(user_tag_arr)
                x = 1
                label_valid.append(x)  # positive sample label: 1

            # negative samples
            temp_tag_list = list(set(valid_tag_list) - set(positive_tag_list)-set(qid_embed_dict[self.user_id]))
            negative_tag_list = random.sample(temp_tag_list, 5 * len(positive_tag_list))
            for tag in negative_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_valid.append(user_tag_arr)
                x = 0
                label_valid.append(x)  # negative sample label: 0

            feature_valid = np.array(feature_valid)
            label_valid = np.array(label_valid)

            mlp_input = torch.FloatTensor(feature_valid)
            mlp_label = torch.FloatTensor(label_valid)

        if x == "test":
            testF = open('./'+modelPath+'/testBertLstmMlp.dat', "a")
            testF.write(f"# query {self.user_num + 1}")

            testF2 = open('./'+modelPath+'/testBertLstmMlp2.dat', "a")
            testF2.write(f"# query {self.user_num + 1}")

            feature_test = []
            label_test = []
            # positive samples
            positive_tag_list = sorted(list(set(qid_test_dict[self.user_id])-set(qid_embed_dict[self.user_id])-set(qid_train_dict[self.user_id])))
            for tag in positive_tag_list:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_test.append(user_tag_arr)
                x = 1
                label_test.append(x)  # positive sample label: 1

                # write qid test file
                Str = f"\n{x} {'qid'}:{self.user_num + 1}"
                testF2.write(Str)

                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                testF.write(Str)

            # negative samples
            negative_tag_list = list(set(test_tag_list) - set(positive_tag_list)-set(qid_embed_dict[self.user_id])-set(qid_train_dict[self.user_id]))
            for tag in negative_tag_list:  # negative samples
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                feature_test.append(user_tag_arr)
                x = 0
                label_test.append(x)  # negative sample label: 0

                # write qid test file
                Str = f"\n{x} {'qid'}:{self.user_num + 1}"
                testF2.write(Str)

                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                testF.write(Str)

            testF.write("\n")
            testF2.write("\n")
            testF.close()
            testF2.close()

            #print(feature_test[0])
            #print(type(feature_test[0]))
            #print(len(feature_test[0]))

            feature_test = np.array(feature_test)
            label_test = np.array(label_test)

            #print(feature_test)
            #print(type(feature_test))
            #print(len(feature_test))

            mlp_input = torch.FloatTensor(feature_test)
            mlp_label = torch.FloatTensor(label_test)

        mlp_hidden = self.fc1(mlp_input)
        mlp_relu = self.relu(mlp_hidden)
        mlp_output = self.fc2(mlp_relu)
        mlp_output = self.sigmoid(mlp_output)
        return mlp_output, mlp_label


def all_user():
    user_num = 0
    for user_id in tqdm(user_list[:100]):
        try:
            each_user(user_num, user_id)
        except:
            continue
        user_num += 1


def each_user(user_num, user_id):
    # model, criterion, optimizer
    model = LstmMlp(user_num, user_id, 768, 30)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.22)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, verbose=True)

    '''
    model.eval()
    label_pred, label_test = model("test")

    before_train = criterion(label_pred.squeeze(), label_test)
    print("test loss before training", before_train.item())
    '''
    # train the model
    model.train()
    epoch = 90

    for epoch in range(epoch):
        # train process-----------------------------------
        optimizer.zero_grad()

        # forward pass
        label_pred, label_train = model("train")

        # compute loss
        loss = criterion(label_pred.squeeze(), label_train)

        # backward pass
        loss.backward()
        optimizer.step()

        # validate process----------------------------------
        optimizer.zero_grad()
        label_pred, label_valid = model("validation")
        val_loss = criterion(label_pred.squeeze(), label_valid)
        scheduler.step(val_loss)

    # evaluation
    model.eval()
    label_pred, label_test = model("test")

    preF = open(modelPath+'/preBertLstmMlp.txt', "a")
    #preF.write(f"# query {user_num + 1}\n")
    spe_user_pre = label_pred.detach().numpy().tolist()
    for tag_pre in spe_user_pre:
        preF.write(f"{tag_pre[0]}\n")
    preF.close()

    print(label_pred.squeeze())
    print(label_test)
    after_train = criterion(label_pred.squeeze(), label_test)
    print("test loss after train", after_train.item())


if __name__ == "__main__":
    all_user()