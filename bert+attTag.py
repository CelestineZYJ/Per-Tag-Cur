# This script is a pytorch dataset returning output after feature engineering
#  -*- coding: utf-8 -*-
import re
import json
import torch
import numpy as np
from tqdm import tqdm
#print(torch.cuda.is_available())
#print(print(torch.__version__))


class MultiheadSelfAttention(torch.nn.Module):
    """
    Multi-headed self attention
    """
    def __init__(
            self,
            input_dim,
            embed_dim,  # q,k,v have the same dimension here
            num_heads=1,  # By default, we use single head
                 ):
        super().__init__()
        self.q_proj = torch.nn.Linear(input_dim, embed_dim)
        self.k_proj = torch.nn.Linear(input_dim, embed_dim)
        self.v_proj = torch.nn.Linear(input_dim, embed_dim)
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    def forward(self, x):
        # x: (Time, Batch_Size, Channel)
        x = torch.unsqueeze(x, 1)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.attention(q, k, v)
        attn_output = torch.squeeze(attn_output, 1)
        maxpool = torch.nn.MaxPool2d(kernel_size=(len(attn_output), 1))
        attn_output = maxpool(attn_output)
        return attn_output  # (Time, Batch_Size, embed_dim)


class Mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embed_size):
        super(Mlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.fc1 = torch.nn.Linear(self.input_size*2, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.attention = MultiheadSelfAttention(self.input_size, self.embed_size)

    def forward(self, user_feature, hashtag_feature):
        hashtag_modeling = self.attention(hashtag_feature)
        x = hashtag_modeling
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
                hashtag_feature.append(self.dict[text])
            user_feature = torch.FloatTensor(user_feature)
            hashtag_feature = torch.FloatTensor(hashtag_feature)
        if self.data_split == 'Valid':
            try:
                for text in self.train_text_per_hashtag[hashtag]:
                    hashtag_feature.append(self.dict[text])
            except:
                pass

            for text in list((set(self.valid_text_per_hashtag[hashtag])-set(self.valid_text_per_user[user]))):
                hashtag_feature.append(self.dict[text])

            user_feature = torch.FloatTensor(user_feature)
            hashtag_feature = torch.FloatTensor(hashtag_feature)
        if self.data_split == 'Test':
            try:
                for text in self.train_text_per_hashtag[hashtag]:
                    hashtag_feature.append(self.dict[text])
            except:
                pass
            for text in list(set(self.test_text_per_hashtag[hashtag])-set(self.test_text_per_user[user])):
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
        if self.data_split == 'Valid':
            trainF = open(self.train_file, encoding='utf-8')
            validF = open(self.valid_file, encoding='utf-8')
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
            for line in validF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.valid_text_per_user.setdefault(user, [])
                self.valid_text_per_user[user].append(text)
                self.valid_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    self.valid_hashtag_list.add(hashtag)
                    self.valid_text_per_hashtag.setdefault(hashtag, [])
                    self.valid_text_per_hashtag[hashtag].append(text)
                    self.valid_hashtag_per_user[user].add(hashtag)

            trainF.close()
            validF.close()
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
                neg_hashtag = list(set(self.train_hashtag_list) - set(self.train_hashtag_per_user[user]))
                num = len(neg_hashtag)
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                    for i in range(self.neg_sampling):
                        j = np.random.randint(num)
                        self.user_hashtag.append((user, neg_hashtag[j]))
                        self.label.append(0)
        if self.data_split == 'Valid':
            for user in self.user_list:
                pos_hashtag = list(set(self.valid_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                neg_hashtag = list(set(self.valid_hashtag_list) - set(self.valid_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                for hashtag2 in neg_hashtag:
                    self.user_hashtag.append((user, hashtag2))
                    self.label.append(0)
        if self.data_split == 'Test':
            labelF = open('./tBertAttTag/testBertAttTag.dat', "a")
            for index, user in enumerate(self.user_list):
                labelF.write(f"# query {index}\n")
                pos_hashtag = list(set(self.test_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                neg_hashtag = list(set(self.test_hashtag_list) - set(self.test_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                    labelF.write(f"{1} qid:{index}\n")
                for hashtag2 in neg_hashtag:
                    self.user_hashtag.append((user, hashtag2))
                    self.label.append(0)
                    labelF.write(f"{0} qid:{index}\n")
            labelF.close()

    def load_tensor_dict(self):
        raise NotImplementedError


# read files
with open('./tData/embeddings.json', 'r') as f:
    text_emb_dict = json.load(f)

with open("tData/userList.txt", "r") as f:
    x = f.readlines()[0]
    user_list = re.findall(r"['\'](.*?)['\']", str(x))


train_file = './tData/train.csv'
valid_file = './tData/valid.csv'
test_file = './tData/test.csv'


def cal_all_pair():
    train_dataset = ScratchDataset(data_split='Train', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    valid_dataset = ScratchDataset(data_split='Valid', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))


    # model, criterion, optimizer
    model = Mlp(768, 30, 100)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # , momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, verbose=True)

    # train the model
    model.train()
    epoch = 10

    for epoch in range(epoch):
        for i in tqdm(range(len(train_dataset))):
            train_hashtag_feature, train_label = train_dataset[i]

            # train process-----------------------------------
            optimizer.zero_grad()

            # forward pass
            pred_label = model(train_hashtag_feature)
            #print(pred_label)

            # compute loss
            #print(train_label)
            loss = criterion(pred_label, train_label)

            #print("Epoch {}: train loss: {}".format(epoch, loss.item()))

            # backward pass
            loss.backward()
            optimizer.step()

            #'''
            # validate process----------------------------------
            try:
                valid_hashtag_feature, valid_label = valid_dataset[i]
                optimizer.zero_grad()
                pred_label = model(valid_hashtag_feature)
                val_loss = criterion(pred_label, valid_label)
                scheduler.step(val_loss)
            except:
                pass
            #'''

    # evaluation
    model.eval()
    fr = open("./tBertAttTag/testBertAttTag.dat", 'r')
    fw = open("./tBertAttTag/testBertAttTag2.dat", 'w')
    lines = fr.readlines()
    lines = [line.strip() for line in lines if line[0] != '#']
    preF = open('./tBertAttTag/preBertAttTag.txt', "a")
    last_user = lines[0][6:]
    for i in tqdm(range(len(test_dataset))):
        test_hashtag_feature, test_label = test_dataset[i]
        try:
            pred_label = model(test_hashtag_feature)
        except:
            continue
        print(pred_label)
        print(test_label)
        preF.write(f"{pred_label.detach().numpy().tolist()[0]}\n")
        after_train = criterion(pred_label, test_label)
        print("test loss after train", after_train.item())
    preF.close()


cal_all_pair()