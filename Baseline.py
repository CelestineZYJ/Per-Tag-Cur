import re
import time as t

import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(int(t.time()))


def get_hashtag(text):
    # 提取hashtag，没有的话返回[],有的话返回["tag1",'tag2']
    h = re.findall(r"#[^#]+#", text)
    h = [i[1:-1].strip() for i in h]
    h = list(set(h))
    return h


def random_rec(dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['hashtag'].loc[np.random.randint(100000)])
    return tag_rec


def popular_rec(dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['hashtag'].loc[i])
    return tag_rec


def latest_rec(user, dataframe):
    tag_rec = []
    dataframe = dataframe.sort_values(by=['time'], ascending=False)  # time
    user_weibo = ''.join(dataframe[dataframe['user_id'] == user]['content'])
    tags = get_hashtag(user_weibo)
    tag_rec = set()
    for tag in tags:
        tag_rec.add(tag)
        if(len(tag_rec) > 5):
            break
    return list(tag_rec)


def eval_rec(dataframe1, dataframe2, dataframe3):
    user_train_df = dataframe1.drop(
        ['weibo_id', 'time', 'content', 'hashtag'], axis=1)
    user_set = set(list(user_train_df['user_id']))
    print(len(user_set))
    user_test_df = dataframe2.drop(['weibo_id', 'time', 'hashtag'], axis=1)

    random_tag_list = random_rec(dataframe3)
    popular_tag_list = popular_rec(dataframe3)

    success = 0
    for user in tqdm(user_set,desc='Eval latest: '):                         # 重复出现的user要记得筛掉
        latest_tag_list = latest_rec(user, dataframe1)
        user_weibo = ''.join(
            user_test_df[user_test_df['user_id'] == user]['content'])
        for tag in latest_tag_list:
            if tag in get_hashtag(user_weibo):
                success += 1
    print("latest recommendation: " + str(success / len(user_set)))

    success = 0
    for user in tqdm(user_set,desc='Eval random: '):
        user_weibo = ''.join(
            user_test_df[user_test_df['user_id'] == user]['content'])
        for tag in random_tag_list:
            if tag in get_hashtag(user_weibo):
                success += 1
    print("random recommendation: "+str(success/len(user_set)))
    success = 0
    for user in tqdm(user_set,desc='Eval popular: '):
        user_weibo = ''.join(
            user_test_df[user_test_df['user_id'] == user]['content'])
        for tag in popular_tag_list:
            if tag in get_hashtag(user_weibo):
                success += 1
    print("popularity recommendation: " + str(success / len(user_set)))


if __name__ == "__main__":
    train_path = "./data/trainSet.csv"
    test_path = "./data/testSet.csv"
    freq_tag_path = "./data/countTrainTag.csv"

    train_df = pd.read_csv(train_path, delimiter='\t', encoding="utf-8")
    test_df = pd.read_csv(test_path, delimiter='\t', encoding="utf-8")
    freq_tag_df = pd.read_csv(freq_tag_path, delimiter='\t', encoding="utf-8")
    eval_rec(train_df, test_df, freq_tag_df)
