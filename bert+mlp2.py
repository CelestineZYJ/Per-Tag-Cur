# This script is a pytorch dataset returning output after feature engineering
#  -*- coding: utf-8 -*-
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from Data.scratch_dataset import my_collate
import torch.utils.data as data

# print(torch.cuda.is_available())
# print(print(torch.__version__))

dataPath = 't'
encoderPath = 'Bert'
secondLayer = ''
classifierPath = 'Mlp'


class Mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Mlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size * 2, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, sign, user_features, user_lens, hashtag_features, hashtag_lens):
        if sign == 'Train':
            user_embeds = self.user_modeling(user_features, user_lens)
            hashtag_embeds = self.hashtag_modeling(hashtag_features, hashtag_lens)
            x = torch.cat((user_embeds, hashtag_embeds), dim=1)

        if sign == 'Test':
            user_modeling = torch.mean(user_features, 0)
            hashtag_modeling = torch.mean(hashtag_features, 0)
            x = torch.cat((user_modeling, hashtag_modeling), 0)

        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

    def user_modeling(self, user_features, user_lens):
        outputs = []
        for user_feature, user_len in zip(user_features, user_lens):
            outputs.append(torch.mean(user_feature[:user_len.item()], dim=0))
        outputs = torch.stack(outputs)
        return outputs

    def hashtag_modeling(self, hashtag_features, hashtag_lens):
        outputs = []
        for hashtag_feature, hashtag_len in zip(hashtag_features, hashtag_lens):
            outputs.append(torch.mean(hashtag_feature[:hashtag_len.item()], dim=0))
        outputs = torch.stack(outputs)
        return outputs


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
            user_feature.append(self.get_feature(self.dict, text))
        # hashtag modeling(train embedding+test others' embedding)
        if self.data_split == 'Train':
            for text in self.train_text_per_hashtag[hashtag]:
                hashtag_feature.append(self.get_feature(self.dict, text))

        if self.data_split == 'Valid':
            try:
                for text in self.train_text_per_hashtag[hashtag]:
                    hashtag_feature.append(self.get_feature(self.dict, text))
            except:
                pass

            for text in list(set(self.valid_text_per_hashtag[hashtag]) - set(self.valid_text_per_user[user])):
                hashtag_feature.append(self.get_feature(self.dict, text))

        if self.data_split == 'Test':
            try:
                for text in self.train_text_per_hashtag[hashtag]:
                    hashtag_feature.append(self.get_feature(self.dict, text))
            except:
                pass

            for text in list(set(self.test_text_per_hashtag[hashtag]) - set(self.test_text_per_user[user])):
                hashtag_feature.append(self.dict[text])

        user_feature = torch.FloatTensor(user_feature)
        hashtag_feature = torch.FloatTensor(hashtag_feature)

        return user_feature, hashtag_feature, torch.FloatTensor([self.label[idx]])

    def get_feature(self, dict, key):
        return dict[key]

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
                neg_hashtag = list(set(self.valid_hashtag_list) - set(self.valid_hashtag_per_user[user]) - set(
                    self.train_hashtag_per_user[user]))
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                for hashtag2 in neg_hashtag:
                    self.user_hashtag.append((user, hashtag2))
                    self.label.append(0)
        if self.data_split == 'Test':
            labelF = open('./'+dataPath+encoderPath+secondLayer+classifierPath+'/test'+encoderPath+secondLayer+classifierPath+'.dat', "a")
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
with open('./'+dataPath+'Data/embeddings.json', 'r') as f:
    text_emb_dict = json.load(f)

with open('./'+dataPath+'Data/userList.txt', "r") as f:
    x = f.readlines()[0]
    user_list = re.findall(r"['\'](.*?)['\']", str(x))

train_file = './'+dataPath+'Data/train.csv'
valid_file = './'+dataPath+'Data/valid.csv'
test_file = './'+dataPath+'Data/test.csv'


def cal_all_pair():
    train_dataset = ScratchDataset(data_split='Train', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    #valid_dataset = ScratchDataset(data_split='Valid', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)

    train_dataloader = data.DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=my_collate, num_workers=0)
    # model, criterion, optimizer
    model = Mlp(768, 30)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # , momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5000, threshold=0.0001, threshold_mode='rel', cooldown=0, verbose=True)

    if torch.cuda.is_available():
        model.cuda()

    # train the model
    model.train()
    epoch = 30

    for epoch in range(epoch):
        for train_user_features, train_user_lens, train_hashtag_features, train_hashtag_lens, labels in tqdm(train_dataloader):
            if torch.cuda.is_available():
                train_user_features = train_user_features.cuda()
                train_user_lens = train_user_lens.cuda()
                train_hashtag_features = train_hashtag_features.cuda()
                train_hashtag_lens = train_hashtag_lens.cuda()
                labels = labels.cuda()

            # train process-----------------------------------
            optimizer.zero_grad()

            # forward pass
            pred_labels = model('Train', train_user_features, train_user_lens, train_hashtag_features, train_hashtag_lens)

            # compute loss
            loss = criterion(pred_labels, labels.reshape(-1, 1))

            #print("Epoch {}: train loss: {}".format(epoch, loss.item()))

            # backward pass
            loss.backward()
            optimizer.step()

            # # validate process----------------------------------
            # try:
            #     valid_user_feature, valid_hashtag_feature, valid_label = valid_dataset[i]
            #     optimizer.zero_grad()
            #     pred_label = model(valid_user_feature, valid_hashtag_feature)
            #     val_loss = criterion(pred_label, valid_label)
            #     scheduler.step(val_loss)
            # except:
            #     pass

    # evaluation
    model.eval()
    fr = open('./'+dataPath+encoderPath+secondLayer+classifierPath+'/test'+encoderPath+secondLayer+classifierPath+'.dat', 'r')
    fw = open('./'+dataPath+encoderPath+secondLayer+classifierPath+'/test'+encoderPath+secondLayer+classifierPath+'2.dat', 'w')
    lines = fr.readlines()
    lines = [line.strip() for line in lines if line[0] != '#']
    preF = open('./'+dataPath+encoderPath+secondLayer+classifierPath+'/pre'+encoderPath+secondLayer+classifierPath+'.txt', "a")
    last_user = lines[0][6:]
    print('# query 0', file=fw)
    for i in tqdm(range(len(test_dataset))):
        line = lines[i]
        test_user_feature, test_hashtag_feature, test_label = test_dataset[i]
        test_user_feature = test_user_feature.cuda()
        test_hashtag_feature = test_hashtag_feature.cuda()
        test_label = test_label.cuda()

        user = line[6:]
        if (user == last_user):
            pass
        else:
            print('# query ' + user, file=fw)
            last_user = user

        try:
            pred_label = model('Test', test_user_feature, 0, test_hashtag_feature, 0)
            print(line, file=fw)
        except:
            print("no test")
            continue

        print(pred_label)
        print(test_label)

        pred_label = pred_label.cpu().detach().numpy().tolist()[0]
        preF.write(f"{pred_label}\n")
        after_train = criterion(pred_label, test_label)
        print("test loss after train", after_train.item())

    preF.close()


cal_all_pair()