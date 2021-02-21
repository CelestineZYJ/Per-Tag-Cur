import pandas as pd
import numpy as np
import torch
import json
import re
from tqdm import tqdm
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator

hashtag_to_id = {}
hashtag_id = 1
user_to_id = {}
user_id = 1


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
            neg_sampling=0,
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
        global hashtag_to_id
        global user_to_id
        global hashtag_id
        global user_id
        user, hashtag = self.user_hashtag[idx]

        try:
            temp = user_to_id[user]
        except:
            user_to_id[user] = user_id
            user_id = user_id + 1

        try:
            temp = hashtag_to_id[hashtag]
        except:
            hashtag_to_id[hashtag] = hashtag_id
            hashtag_id = hashtag_id + 1

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
                hashtag_feature.append(self.get_feature(self.dict, text))
            if len(hashtag_feature) == 0:
                hashtag_feature.append([0.] * 768)

        user_feature = torch.FloatTensor(user_feature)
        hashtag_feature = torch.FloatTensor(hashtag_feature)

        return user_feature, hashtag_feature, torch.FloatTensor([self.label[idx]]), user_to_id[user], hashtag_to_id[hashtag]

    def get_feature(self, dict, key):
        try:
            return dict[key]
        except:
            return [0.]*768

    def __len__(self):
        return len(self.label)

    # cal user modeling and hashtag modeling
    def process_data_file(self):
        with open('./data/hashtag_fake_split_weibo_update.csv', encoding='utf-8') as f:
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
                    for i in range(15):
                        j = np.random.randint(num)
                        self.user_hashtag.append((user, neg_hashtag[j]))
                        self.label.append(0)
        if self.data_split == 'Test':
            labelF = open('./data/testNcf.dat', "a")
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

                # for i in range(num):
                #     self.user_hashtag.append((user, neg_hashtag[i]))
                #     self.label.append(0)
                #     labelF.write(f"{0} qid:{index}\n")

                # for hashtag2 in neg_hashtag:
                #     self.user_hashtag.append((user, hashtag2))
                #     self.label.append(0)
                #     labelF.write(f"{0} qid:{index}\n")
            labelF.close()

    def load_tensor_dict(self):
        raise NotImplementedError


# read files
# with open('./data/embeddings.json', 'r') as f:
#     text_emb_dict = json.load(f)
text_emb_dict = {}

with open('./data/userList.txt', "r") as f:
    x = f.readlines()[0]
    user_list = re.findall(r"['\'](.*?)['\']", str(x))

train_file = './data/train.csv'
test_file = './data/test.csv'

train_dataset = ScratchDataset(data_split='Train', user_list=user_list, train_file=train_file, valid_file=None, test_file=test_file, dict=text_emb_dict)
test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=None, test_file=test_file, dict=text_emb_dict)


with open('./data/train_rating2.dat', 'w') as f:
    for user_feature, hashtag_feature, label, user, hashtag in tqdm(train_dataset):
        f.writelines(str(user)+'::'+str(hashtag)+'::'+str(1)+'::'+'1'+'\n')
f.close()
'''
import json 

json_user = json.dumps(user_to_id)
f = open('./data/user_to_id.json', 'w')
f.write(json_user)
f.close()

json_hashtag = json.dumps(hashtag_to_id)
f = open('./data/hashtag_to_id.json', 'w')
f.write(json_hashtag)
f.close()


with open('./data/user_to_id.json', 'r') as f:
    user_to_id = json.load(f)

with open('./data/hashtag_to_id.json', 'r') as f:
    hashtag_to_id = json.load(f)
'''
gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 200,
              'batch_size': 256,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-2,
              'num_users': 589,
              'num_items': 42762,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': False,
              'device_id': 1,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 200,
              'batch_size': 256,
              'optimizer': 'adam',
              'adam_lr': 1e-2,
              'num_users': 589,
              'num_items': 42762,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 1,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch99_HR0.6391_NDCG0.2852.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch':100,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users':589,# 589,
                'num_items': 42762,#42762,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 1,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch99_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch99_HR0.5606_NDCG0.2463.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

# Load Data
ml1m_dir = 'data/train_rating2.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data
# Specify the exact model
#config = gmf_config
#engine = GMFEngine(config)
config = mlp_config
engine = MLPEngine(config)
# config = neumf_config
# engine = NeuMFEngine(config)
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)


# # test
# fr = open('./data/testNcf.dat', 'r')
# fw = open('./data/testNcf2.dat', 'w')
# preF = open('./data/preNcf.txt', 'a')

# lines = fr.readlines()
# lines = [line.strip() for line in lines if line[0] != '#']
# preF = open('./data/preNcf.txt', "a")
# last_user = lines[0][6:]
# print('# query 0', file=fw)
# with torch.no_grad():
#     for i in tqdm(range(len(test_dataset))):
#         line = lines[i]
#         test_user_feature, test_hashtag_feature, test_label, test_user, test_hashtag = test_dataset[i]
#         test_user = torch.LongTensor([test_user])
#         test_hashtag = torch.LongTensor([test_hashtag])

#         user = line[6:]
#         if (user == last_user):
#             pass
#         else:
#             print('# query ' + user, file=fw)
#             last_user = user
#         pred_label = engine.test(test_user, test_hashtag)
#         print(line, file=fw) 
            
#         # except:
#             # print(test_hashtag)
#             # continue
#         print(pred_label)
#         print(test_label)
#         pred_label = pred_label.cpu().detach().numpy().tolist()[0]
#         preF.write(f"{pred_label}\n")
# preF.close()


