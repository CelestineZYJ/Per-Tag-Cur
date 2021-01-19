# This script is a pytorch dataset returning output after feature engineering
#  -*- coding: utf-8 -*-
import re
import json
import torch
import numpy as np
from tqdm import tqdm


class Mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Mlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size*2, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_feature, hashtag_feature):
        # user modeling
        user_modeling = torch.mean(user_feature, 0)
        # hashtag modeling
        hashtag_modeling = torch.mean(hashtag_feature, 0)
        x = torch.cat((user_modeling, hashtag_modeling), 0)
        #print(x)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


class ScratchDataset(torch.utils.data.Dataset):
    """
    Return (all tensors of user,  all tensors of hashtag, label)
    """
    def __init__(
            self,
            data_split,
            user_list,
            train_file,
            valid_file,
            test_file,
            dict,  # you need to implement load dict of tensors by yourself
            neg_sampling=5,
            ):
        """
        user_list: users occurs in both train, valid and test (which we works on)
        data_file: format of 'twitter_text    user     hashtag1     hashtag2     ...'
        data_split: train/val/test
        """
        self.data_split = data_split
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.neg_sampling = neg_sampling
        self.dict = dict
        self.user_list = user_list

        self.train_hashtag_list = set()
        self.train_hashtag_per_user = {}
        self.train_text_per_user = {}
        self.train_text_per_hashtag = {}

        self.valid_hashtag_list = set()
        self.valid_hashtag_per_user = {}
        self.valid_text_per_user = {}
        self.valid_text_per_hashtag = {}

        self.test_hashtag_list = set()
        self.test_hashtag_per_user = {}
        self.test_text_per_user = {}
        self.test_text_per_hashtag = {}

        self.user_hashtag = []
        self.label = []

        self.process_data_file()
        self.create_dataset()

    def __getitem__(self, idx):
        user, hashtag = self.user_hashtag[idx]
        user_feature, hashtag_feature = [], []
        # user modeling(always train embedding)
        for text in self.train_text_per_user[user]:
            user_feature.append(self.dict[text])

        # hashtag modeling(train embedding+test others' embedding)
        if self.data_split == 'Train':
            for text in self.train_text_per_hashtag[hashtag]:
                try:
                    hashtag_feature.append(self.dict[text])
                except:
                    #print("found no: "+text)
                    continue
            user_feature = torch.FloatTensor(user_feature)
            hashtag_feature = torch.FloatTensor(hashtag_feature)
        if self.data_split == 'Test':
            for text in self.train_text_per_hashtag[hashtag]:
                hashtag_feature.append(self.dict[text])
            for text in (self.test_text_per_hashtag[hashtag]-self.test_text_per_user[user]):
                hashtag_feature.append(self.dict[text])
            user_feature = torch.FloatTensor(user_feature)
            hashtag_feature = torch.FloatTensor(hashtag_feature)
        return user_feature, hashtag_feature, torch.FloatTensor([self.label[idx]])

    def __len__(self):
        return len(self.label)

    # cal user modeling and hashtag modeling
    def process_data_file(self):
        if self.data_split == 'Train':
            trainF = open(self.train_file, encoding='utf-8')
            for line in trainF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.train_text_per_user.setdefault(user, [])
                self.train_text_per_user[user].append(text)
                self.train_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    self.train_hashtag_list.add(hashtag)
                    self.train_text_per_hashtag.setdefault(hashtag, [])
                    self.train_text_per_hashtag[hashtag].append(text)
                    self.train_hashtag_per_user[user].add(hashtag)
            f.close()

        if self.data_split == 'Test':
            trainF = open(self.train_file, encoding='utf-8')
            testF = open(self.test_file, encoding='utf-8')
            for line in trainF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.train_text_per_user.setdefault(user, [])
                self.train_text_per_user[user].append(text)
                self.train_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    self.train_hashtag_list.add(hashtag)
                    self.train_text_per_hashtag.setdefault(hashtag, [])
                    self.train_text_per_hashtag[hashtag].append(text)
                    self.train_hashtag_per_user[user].add(hashtag)
            for line in testF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.test_text_per_user.setdefault(user, [])
                self.test_text_per_user[user].append(text)
                self.test_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    self.test_hashtag_list.add(hashtag)
                    self.test_text_per_hashtag.setdefault(hashtag, [])
                    self.test_text_per_hashtag[hashtag].append(text)
                    self.test_hashtag_per_user[user].add(hashtag)
            trainF.close()
            testF.close()

    def create_dataset(self):
        """
        Do positive and negative sampling here
        """
        if self.data_split == 'Train':
            for user in self.user_list:
                pos_hashtag = self.train_hashtag_per_user[user]
                neg_hashtag = list(self.train_hashtag_list - self.train_hashtag_per_user[user])
                num = len(neg_hashtag)
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                    for i in range(self.neg_sampling):
                        j = np.random.randint(num)
                        self.user_hashtag.append((user, neg_hashtag[j]))
                        self.label.append(0)
        if self.data_split == 'Test':
            for user in self.user_list:
                pos_hashtag = self.test_hashtag_per_user[user] - self.train_hashtag_per_user[user]
                neg_hashtag = self.test_hashtag_list - self.test_hashtag_per_user[user] - self.train_hashtag_per_user[user]
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                    for hashtag2 in neg_hashtag:
                        self.user_hashtag.append((user, hashtag2))
                        self.label.append(0)

    def load_tensor_dict(self):
        raise NotImplementedError


# read files
with open('./demo2/embeddings.json', 'r') as f:
    text_emb_dict = json.load(f)

with open("demo2/userList.csv", "r") as f:
    x = f.readlines()[0]
    user_list = re.findall(r"['\'](.*?)['\']", str(x))


train_file = './demo2/train.csv'
valid_file = './demo2/valid.csv'
test_file = './demo2/test.csv'


def cal_all_pair():
    train_dataset = ScratchDataset(data_split='Train', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    #valid_dataset = ScratchDataset(data_split='Valid', user_list=user_list, train_file=train_file, valid_file=vaid_file, test_file=test_file, dict=text_emb_dict)
    test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)

    # model, criterion, optimizer
    model = Mlp(768, 30)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)  # , momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, verbose=True)

    # train the model
    model.train()
    epoch = 10

    for epoch in range(epoch):
        print(epoch)
        for i in tqdm(range(len(train_dataset))):
            train_user_feature, train_hashtag_feature, train_label = train_dataset[i]

            # train process-----------------------------------
            optimizer.zero_grad()

            # forward pass
            try:
                pred_label = model(train_user_feature, train_hashtag_feature)
                # print(pred_label)
            except:
                continue

            # compute loss
            # print(train_label)
            try:
                loss = criterion(pred_label, train_label)
            except:
                tt = 1

            #print("Pair "+str(i)+": ")
            #print("Epoch {}: train loss: {}".format(epoch, loss.item()))

            # backward pass
            loss.backward()
            optimizer.step()

            '''
            # validate process----------------------------------
            valid_user_feature, valid_hashtag_feature, valid_label = valid_dataset[i]
            optimizer.zero_grad()
            pred_label = model(valid_user_feature, valid_hashtag_feature)
            val_loss = criterion(pred_label.squeeze(), valid_label)
            scheduler.step(val_loss)
            '''
    # evaluation
    model.eval()
    for i in range(len(test_dataset)):
        test_user_feature, test_hashtag_feature, test_label = test_dataset[i]
        pred_label = model(test_user_feature, test_hashtag_feature)
        print(pred_label.squeeze())
        print(test_label)
        after_train = criterion(pred_label.squeeze(), test_label)
        print("Pair " + str(i) + ": ")
        print("test loss after train", after_train.item())


cal_all_pair()





