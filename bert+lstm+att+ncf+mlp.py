# This script is a pytorch dataset returning output after feature engineering
#  -*- coding: utf-8 -*-
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from Data.scratch_dataset import my_collate
from Modules.utils import weighted_class_bceloss
import torch.utils.data as data
import torch.nn.functional as F
from Modules.self_attention import MultiheadSelfAttention
from Modules.neumf import Neumf

torch.manual_seed(2021)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2021)

dataPath = 'w'
encoderPath = 'Bert'
secondLayer = 'LstmAttNcf'
classifierPath = 'Mlp'
indexPath = ''

config = {'user_modeling': 'mean',
                'hashtag_modeling': 'mean',
                'interaction_modeling': 'ncf',
                'num_users': 597,
                'num_items': 23733,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
    }


class Mlp(torch.nn.Module):
    def __init__(self, config, input_size, hidden_size):
        super(Mlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(self.input_size * 2, self.hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(num_features=self.hidden_size)
        # self.fc2 = torch.nn.Linear(self.input_size, self.hidden_size)
        # self.bn2 = torch.nn.BatchNorm1d(num_features=self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)
        self.lstm = torch.nn.LSTM(768, 768)
        self.sigmoid = torch.nn.Sigmoid()
        self.attention = MultiheadSelfAttention(input_dim=768, embed_dim=768)
        self.interaction = Neumf(config)

    def forward(self, sign, user_features, user_lens, hashtag_features, hashtag_lens, users, items):
        user_embeds = self.user_modeling(user_features, user_lens)
        hashtag_embeds = self.hashtag_modeling(hashtag_features, hashtag_lens, user_embeds=user_embeds)
        ncf_embeds = self.interaction_modeling(users, items)
        if config['user_modeling'] == 'None' and config['hashtag_modeling'] == 'None':
            x = ncf_embeds
        else:
            x = torch.cat((user_embeds, hashtag_embeds, ncf_embeds), dim=1)

        x = self.relu(self.bn1(self.fc1(x)))
        # x = self.relu(self.bn2(self.fc2(x)))
        output = self.fc3(x)

        output = self.sigmoid(output)
        return output

    def user_modeling(self, user_features, user_lens):
        if config['user_modeling'] == 'lstm':
            inputs = torch.nn.utils.rnn.pack_padded_sequence(user_features, user_lens, batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(inputs)
            return h[-1]
        elif config['user_modeling'] == 'mean':
            outputs = []
            for user_feature, user_len in zip(user_features, user_lens):
                outputs.append(torch.mean(user_feature[:user_len.item()], dim=0))
            outputs = torch.stack(outputs)
            return outputs

    def hashtag_modeling(self, hashtag_features, hashtag_lens, config='mean', user_embeds=None):
        if config == 'attn':
            masks = torch.where(hashtag_features[:, :, 0] != 0, torch.Tensor([0.]).cuda(), torch.Tensor([-np.inf]).cuda())
            user_embeds = user_embeds.unsqueeze(1)
            att_weights = (hashtag_features * user_embeds).sum(-1) + masks
            att_weights = F.softmax(att_weights, 1)
            outputs = torch.bmm(hashtag_features.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
        elif config == 'mean':
            outputs = []
            for hashtag_feature, hashtag_len in zip(hashtag_features, hashtag_lens):
                outputs.append(torch.mean(hashtag_feature[:hashtag_len.item()], dim=0))
            outputs = torch.stack(outputs)
        # print(outputs.size()) # torch.Size([128=batch_size, 768])
        return outputs

    def interaction_modeling(self, users, items):
        outputs = self.interaction(users, items)
        print(outputs.size())
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

        self.hashtag_to_id = {}
        self.hashtag_id = 0
        self.user_to_id = {}
        self.user_id = 0
        self.user_hashtag = []
        self.label = []

        self.process_data_file()
        self.create_dataset()

    def __getitem__(self, idx):
        user, hashtag = self.user_hashtag[idx]
        if hashtag in self.hashtag_to_id:
            pass
        else:
            self.hashtag_to_id[hashtag] = self.hashtag_id
            self.hashtag_id += 1

        if user in self.user_to_id:
            pass
        else:
            self.user_to_id[user] = self.user_id
            self.user_id += 1

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
                hashtag_feature.append([0.] * 768)

        if self.data_split == 'Test':
            texts = []
            for sub_hashtag in self.hashtag_split[hashtag]:
                texts += self.test_text_per_hashtag[sub_hashtag]
            for text in list(set(texts) - set(self.test_text_per_user[user])):
                hashtag_feature.append(self.dict[text])
            if len(hashtag_feature) == 0:
                hashtag_feature.append([0.] * 768)

        user_feature = torch.FloatTensor(user_feature)
        hashtag_feature = torch.FloatTensor(hashtag_feature)

        return user_feature, hashtag_feature, torch.FloatTensor([self.label[idx]]), self.user_to_id[user], self.hashtag_to_id[hashtag]

    def get_feature(self, dict, key):
        return dict[key]

    def __len__(self):
        return len(self.label)

    # cal user modeling and hashtag modeling
    def process_data_file(self):
        with open('./'+dataPath+'Data/hashtag_fake_split_weibo_update.csv', encoding='utf-8') as f:
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
                neg_hashtag = list(set(self.valid_hashtag_list) - set(self.valid_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                num = len(neg_hashtag)
                for hashtag in pos_hashtag:
                    self.user_hashtag.append((user, hashtag))
                    self.label.append(1)
                    for i in range(30):
                        j = np.random.randint(num)
                        self.user_hashtag.append((user, neg_hashtag[j]))
                        self.label.append(0)
        if self.data_split == 'Test':
            labelF = open('./'+dataPath+encoderPath+secondLayer+classifierPath+indexPath+'/test'+encoderPath+secondLayer+classifierPath+'.dat', "a")
            for index, user in enumerate(self.user_list):
                labelF.write(f"# query {index}\n")
                pos_hashtag = list(set(self.test_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
                neg_hashtag = list(set(self.test_hashtag_list) - set(self.test_hashtag_per_user[user]) - set(self.train_hashtag_per_user[user]))
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
    train_dataset = ScratchDataset(data_split='Train', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    valid_dataset = ScratchDataset(data_split='Valid', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)
    test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, dict=text_emb_dict)

    train_dataloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=my_collate, num_workers=1)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=512, collate_fn=my_collate, num_workers=1)
    # model, criterion, optimizer
    model = Mlp(config, 768, 256)
    # criterion = torch.nn.BCELoss()
    weights = torch.Tensor([1, 150])
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # , momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=1e-4, min_lr=1e-6)

    if torch.cuda.is_available():
        model = model.cuda()
        weights = weights.cuda()

    # train the model
    epoch = 25
    best_valid_loss = 1e10
    best_epoch = -1

    for epoch in range(epoch):
        num_positive, num_negative = 0., 0.
        num_correct_positive, num_correct_negative = 0, 0
        total_loss = 0.

        model.train()
        for train_user_features, train_user_lens, train_hashtag_features, train_hashtag_lens, labels, users, items in tqdm(train_dataloader):
            if torch.cuda.is_available():
                train_user_features = train_user_features.cuda()
                train_user_lens = train_user_lens.cuda()
                train_hashtag_features = train_hashtag_features.cuda()
                train_hashtag_lens = train_hashtag_lens.cuda()
                labels = labels.cuda()
                users = users.cuda()
                items = items.cuda()

            # train process-----------------------------------
            optimizer.zero_grad()

            # forward pass
            pred_labels = model('Train', train_user_features, train_user_lens, train_hashtag_features, train_hashtag_lens, users, items)

            # compute loss
            loss = weighted_class_bceloss(pred_labels, labels.reshape(-1, 1), weights)
            total_loss += (loss.item() * len(labels))

            for pred_label, label in zip(pred_labels, labels.reshape(-1, 1)):
                if label == 1:
                    num_positive += 1
                    if pred_label > 0.9:
                        num_correct_positive += 1
                else:
                    num_negative += 1
                    if pred_label < 0.9:
                        num_correct_negative += 1

            # backward pass
            loss.backward()
            optimizer.step()

        print('train positive_acc: %f    train negative_acc: %f    train_loss: %f' % \
              ((num_correct_positive / num_positive), (num_correct_negative / num_negative), (total_loss / len(train_dataset))))

        num_positive, num_negative = 0., 0.
        num_correct_positive, num_correct_negative = 0, 0
        total_loss = 0.

        # best_model = Mlp(768, 256)
        # best_model.load_state_dict(torch.load(f'/home/yjzhang/exp/{exp_name}/model_9.pt'))
        # if torch.cuda.is_available():
        #     best_model = best_model.cuda()
        model.eval()
        with torch.no_grad():
            for user_features, user_lens, hashtag_features, hashtag_lens, labels in tqdm(valid_dataloader):
                if torch.cuda.is_available():
                    user_features = user_features.cuda()
                    user_lens = user_lens.cuda()
                    hashtag_features = hashtag_features.cuda()
                    hashtag_lens = hashtag_lens.cuda()
                    labels = labels.cuda()
                pred_labels = model('Train', user_features, user_lens, hashtag_features, hashtag_lens)
                loss = weighted_class_bceloss(pred_labels, labels.reshape(-1, 1), weights)
                total_loss += (loss.item() * len(labels))
                for pred_label, label in zip(pred_labels, labels.reshape(-1, 1)):
                    if label == 1:
                        num_positive += 1
                        if pred_label > 0.9:
                            num_correct_positive += 1
                    else:
                        num_negative += 1
                        if pred_label < 0.9:
                            num_correct_negative += 1

        print('epoch: %d      valid positive_acc: %f   valid negative_acc: %f     valid_loss: %f' % \
              ((epoch + 1), (num_correct_positive / num_positive), (num_correct_negative / num_negative), (total_loss / len(valid_dataset))))
        scheduler.step(total_loss / len(valid_dataset))
        print('learning rate:  %f' % optimizer.param_groups[0]['lr'])

        if total_loss < best_valid_loss:
            best_valid_loss = total_loss
            best_epoch = epoch
            print('Current best!')
            torch.save(model.state_dict(), './'+dataPath+encoderPath+secondLayer+classifierPath+indexPath+'/best_model.pt')
        torch.save(model.state_dict(), './'+dataPath+encoderPath+secondLayer+classifierPath+indexPath+f'/model_{epoch}.pt')

    # test
    best_model = Mlp(config, 768, 256)
    best_model.load_state_dict(torch.load('./'+dataPath+encoderPath+secondLayer+classifierPath+indexPath+f'/model_{best_epoch}.pt'))
    # best_model.load_state_dict(torch.load(f'/home/yjzhang/exp/{exp_name}/best_model.pt'))
    if torch.cuda.is_available():
        best_model = best_model.cuda()
    best_model.eval()
    fr = open('./' + dataPath + encoderPath + secondLayer + classifierPath + indexPath + '/test' + encoderPath + secondLayer + classifierPath + '.dat', 'r')
    fw = open('./' + dataPath + encoderPath + secondLayer + classifierPath + indexPath + '/test' + encoderPath + secondLayer + classifierPath + '2.dat', 'w')
    lines = fr.readlines()
    lines = [line.strip() for line in lines if line[0] != '#']
    preF = open('./' + dataPath + encoderPath + secondLayer + classifierPath + indexPath + '/pre' + encoderPath + secondLayer + classifierPath + '.txt', "a")
    last_user = lines[0][6:]
    print('# query 0', file=fw)
    with torch.no_grad():
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
                pred_label = best_model('Test', test_user_feature.unsqueeze(0), torch.tensor([len(test_user_feature)]), test_hashtag_feature.unsqueeze(0), torch.tensor([len(test_hashtag_feature)]))
                print(line, file=fw)
            except:
                print("no test")
                continue

            print(pred_label)
            print(test_label)

            pred_label = pred_label.cpu().detach().numpy().tolist()[0]
            preF.write(f"{pred_label}\n")
        # after_train = criterion(pred_label, test_label)
        # print("test loss after train", after_train.item())

    preF.close()


if __name__ == '__main__':
    cal_all_pair()
