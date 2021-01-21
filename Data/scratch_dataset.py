# This script is a pytorch dataset returning output after feature engineering
import torch
import numpy as np


class ScratchDataset(torch.utils.data.Dataset):
    """
    Return (all tensors of user,  all tensors of hashtag, label)
    """
    def __init__(
            self,
            user_list,
            data_file,
            dict,  # you need to implement load dict of tensors by yourself
            data_split='Train',
            neg_sampling=5,
            ):
        """
        user_list: users occurs in both train and test (which we works on)
        data_file: format of 'twitter_text    user     hashtag1     hashtag2     ...'
        data_split: train/val/test
        """
        self.data_file = data_file
        self.neg_sampling = neg_sampling
        self.dict = dict
        self.user_list = user_list
        self.hashtag_list = set()
        self.hashtag_per_user = {}
        self.text_per_user = {}
        self.text_per_hashtag = {}
        self.user_hashtag = []
        self.label = []

        self.process_data_file()
        self.create_dataset()

    def __getitem__(self, idx):
        user, hashtag = self.user_hashtag[idx]
        user_feature, hashtag_feature = [], []
        for text in self.text_per_user:
            user_feature.append(self.dict[text])
        for text in self.text_per_hashtag:
            hashtag_feature.append(self.dict[text])
        user_feature = torch.cat(user_feature, dim=0)
        hashtag_feature = torch.cat(hashtag_feature, dim=0)
        return user_feature, hashtag_feature, torch.tensor(self.label[idx])

    def __len__(self):
        return len(self.label)

    def process_data_file(self):
        f = open(self.data_file)
        for line in f:
            l = line.strip('\n').split('\t')
            text, user, hashtags = l[0], l[1], l[2:]
            self.text_per_user.setdefault(user, [])
            self.text_per_user[user].append(text)
            self.hashtag_per_user.setdefault(user, set())
            for hashtag in hashtags:
                self.hashtag_list.add(hashtag)
                self.text_per_hashtag.setdefault(hashtag, [])
                self.text_per_hashtag[hashtag].append(text)
                self.hashtag_per_user[user].add(hashtag)
        f.close()

    def create_dataset(self):
        """
        Do negative sampling here
        """
        for user in self.user_list:
            pos_hashtag = self.hashtag_per_user['user']
            neg_hashtag = list(self.hashtag_list - self.hashtag_per_user['user'])
            num = len(neg_hashtag)
            for hashtag in pos_hashtag:
                self.user_hashtag.append((user, hashtag))
                self.label.append(1)
                for i in range(self.neg_sampling):
                    j = np.random.randint(num)
                    self.user_hashtag.append((user, neg_hashtag[j]))
                    self.label.append(0)

    def load_tensor_dict(self):
        raise NotImplementedError


def my_collate(batch):
    user_features, hashtag_features, labels = [], [], []
    user_len, hashtag_len = [], []
    batch_size = len(batch)
    for user_feature, hashtag_feature, label in batch:
        user_features.append(user_feature)
        hashtag_features.append(hashtag_feature)
        labels.append(label)
        user_len.append(user_feature.shape[0])
        hashtag_len.append(hashtag_feature.shape[0])
    max_user_len, max_hashtag_len = max(user_len), max(hashtag_len)
    for i in range(batch_size):
        user_feature, hashtag_feature = user_features[i], hashtag_features[i]
        user_features[i] = torch.cat((user_feature, torch.zeros(max_user_len-len(user_feature), 768)), dim=0)
        hashtag_features[i] = torch.cat((hashtag_feature, torch.zeros(max_hashtag_len-len(hashtag_feature), 768)), dim=0)
    user_features, hashtag_features = torch.stack(user_features), torch.stack(hashtag_features)
    user_len, hashtag_len, labels = torch.tensor(user_len), torch.tensor(hashtag_len), torch.tensor(labels)
    return user_features, user_len, hashtag_features, hashtag_len, labels
