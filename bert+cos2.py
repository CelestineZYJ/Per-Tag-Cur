# This script is a pytorch dataset returning output after feature engineering
#  -*- coding: utf-8 -*-
import re
import json
import torch
from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm

# print(torch.cuda.is_available())
# print(print(torch.__version__))

dataPath = 't'
encoderPath = 'Bert'
secondLayer = 'Cos'
classifierPath = ''
indexPath = ''


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
        self.hashtag_split = {}

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
            user_feature.append(self.get_feature(self.dict, text))
        # hashtag modeling(train embedding+test others' embedding)

        for sub_hashtag in self.hashtag_split[hashtag]:
            if sub_hashtag in self.train_text_per_hashtag:
                for text in self.train_text_per_hashtag[sub_hashtag]:
                    hashtag_feature.append(self.get_feature(self.dict, text))

        if self.data_split == 'Train':
            if len(hashtag_feature) == 0:
                hashtag_feature.append([0.] * 768)

        if self.data_split == 'Valid':
            texts = []
            for sub_hashtag in self.hashtag_split[hashtag]:
                texts += self.valid_text_per_hashtag[sub_hashtag]
            for text in list(set(texts) - set(self.valid_text_per_user[user])):
                hashtag_feature.append(self.get_feature(self.dict, text))
            if len(hashtag_feature) == 0:
                hashtag_feature.append([-1.0] * 768)

        if self.data_split == 'Test':
            texts = []
            for sub_hashtag in self.hashtag_split[hashtag]:
                texts += self.test_text_per_hashtag[sub_hashtag]
            for text in list(set(texts) - set(self.test_text_per_user[user])):
                hashtag_feature.append(self.dict[text])
            if len(hashtag_feature) == 0:
                hashtag_feature.append([-1.0] * 768)

        user_feature = torch.FloatTensor(user_feature)
        hashtag_feature = torch.FloatTensor(hashtag_feature)

        return user_feature, hashtag_feature, torch.FloatTensor([self.label[idx]])

    def get_feature(self, dict, key):
        return dict[key]

    def __len__(self):
        return len(self.label)

    # cal user modeling and hashtag modeling
    def process_data_file(self):
        with open('./' + dataPath + 'Data/hashtag_fake_split_twitter_update.csv', encoding='utf-8') as f:
            for line in f:
                l = line.strip('\n').strip('\t').split('\t')
                self.hashtag_split[l[0]] = l[1:]
        f.close()

        trainF = open(self.train_file, encoding='utf-8')
        for line in trainF:
            l = line.strip('\n').split('\t')
            text, user, hashtags = l[0], l[1], l[2:]
            self.train_text_per_user.setdefault(user, [])
            self.train_text_per_user[user].append(text)
            self.train_hashtag_per_user.setdefault(user, set())
            for hashtag in hashtags:
                if len(hashtag) == 0:
                    continue
                self.train_hashtag_list.add(hashtag)
                self.train_hashtag_per_user[user].add(hashtag)
                for sub_hashtag in self.hashtag_split[hashtag]:
                    self.train_text_per_hashtag.setdefault(sub_hashtag, [])
                    self.train_text_per_hashtag[sub_hashtag].append(text)
        trainF.close()

        if self.data_split == 'Valid':
            validF = open(self.valid_file, encoding='utf-8')
            for line in validF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.valid_text_per_user.setdefault(user, [])
                self.valid_text_per_user[user].append(text)
                self.valid_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    if len(hashtag) == 0:
                        continue
                    self.valid_hashtag_list.add(hashtag)
                    self.valid_hashtag_per_user[user].add(hashtag)
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        self.valid_text_per_hashtag.setdefault(sub_hashtag, [])
                        self.valid_text_per_hashtag[sub_hashtag].append(text)
            validF.close()

        if self.data_split == 'Test':
            testF = open(self.test_file, encoding='utf-8')
            for line in testF:
                l = line.strip('\n').split('\t')
                text, user, hashtags = l[0], l[1], l[2:]
                self.test_text_per_user.setdefault(user, [])
                self.test_text_per_user[user].append(text)
                self.test_hashtag_per_user.setdefault(user, set())
                for hashtag in hashtags:
                    if len(hashtag) == 0:
                        continue
                    self.test_hashtag_list.add(hashtag)
                    self.test_hashtag_per_user[user].add(hashtag)
                    for sub_hashtag in self.hashtag_split[hashtag]:
                        self.test_text_per_hashtag.setdefault(sub_hashtag, [])
                        self.test_text_per_hashtag[sub_hashtag].append(text)
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
                neg_hashtag = list(set(self.valid_hashtag_list) - set(self.valid_hashtag_per_user[user]) - set(
                    self.train_hashtag_per_user[user]))
                num = len(neg_hashtag)
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                    for i in range(30):
                        j = np.random.randint(num)
                        self.user_hashtag.append((user, neg_hashtag[j]))
                        self.label.append(0)
        if self.data_split == 'Test':
            labelF = open(
                './' + dataPath + encoderPath + secondLayer + classifierPath + indexPath + '/test' + encoderPath + secondLayer + classifierPath + '.dat',
                "a", encoding='utf-8')
            for index, user in enumerate(self.user_list):
                labelF.write(f"# query {index}\n")
                pos_hashtag = list(set(self.test_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                neg_hashtag = list(set(self.test_hashtag_list) - set(self.test_hashtag_per_user[user]) - set(
                    self.train_hashtag_per_user[user]))
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                    labelF.write(f"{1} qid:{index}\n")

                num = len(neg_hashtag)
                for i in range(len(pos_hashtag) * 100):
                    j = np.random.randint(num)
                    self.user_hashtag.append((user, neg_hashtag[j]))
                    self.label.append(0)
                    labelF.write(f"{0} qid:{index}\n")

                # for hashtag2 in neg_hashtag:
                #     self.user_hashtag.append((user, hashtag2))
                #     self.label.append(0)
                #     labelF.write(f"{0} qid:{index}\n")
            labelF.close()

    def load_tensor_dict(self):
        raise NotImplementedError


# read files
with open('./'+dataPath+'Data/embeddings.json', 'r') as f:
    text_emb_dict = json.load(f)

with open('./'+dataPath+'Data/userList.txt', "r") as f:
    x = f.readlines()[0]
    user_list = re.findall(r"['\'](.*?)['\']", str(x))

train_file = './'+dataPath+'Data/train.csv'
valid_file = './'+dataPath+'Data/valid.csv'
test_file = './'+dataPath+'Data/test.csv'


def cal_all_pair():
    test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)

    fr = open('./' + dataPath + encoderPath + secondLayer + classifierPath + '/test' + encoderPath + secondLayer + classifierPath + '.dat', 'r')
    fw = open('./' + dataPath + encoderPath + secondLayer + classifierPath + '/test' + encoderPath + secondLayer + classifierPath + '2.dat', 'w')
    lines = fr.readlines()
    lines = [line.strip() for line in lines if line[0] != '#']
    preF = open('./' + dataPath + encoderPath + secondLayer + classifierPath + '/pre' + encoderPath + secondLayer + classifierPath + '.txt', "a")
    last_user = lines[0][6:]
    print('# query 0', file=fw)
    for i in tqdm(range(len(test_dataset))):
        line = lines[i]
        test_user_feature, test_hashtag_feature, test_label = test_dataset[i]
        user = line[6:]
        if (user == last_user):
            pass
        else:
            print('# query '+user, file=fw)
            last_user = user
        try:
            pred_label = cosine_similarity(torch.mean(test_user_feature, 0), torch.mean(test_hashtag_feature, 0), dim=0)
            print(line, file=fw)
        except:
            continue

        preF.write(f"{pred_label}\n")

    preF.close()


cal_all_pair()
