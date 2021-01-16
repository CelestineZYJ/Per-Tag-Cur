#  -*- coding: utf-8 -*-
import ast
import json
import random
import re
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import rapidjson
import argparse
from tqdm import tqdm


def average_user_tweet(user_list, content_user_df, con_emb_dict):
    user_arr_dict = OrderedDict()
    for user in tqdm(user_list, desc="average_user_tweet"):
        embed_list = []
        content_list = content_user_df[content_user_df['user_id'] == user].content.tolist()[0]
        for content in content_list:
            embed_list.append(con_emb_dict[content])
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list
    return user_arr_dict


def average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict):
    tag_arr_dict = OrderedDict()
    for index, tag in enumerate(tqdm(tag_list, desc="average_hashtag_tweet")):
        embed_list = []
        content_list = content_tag_df[content_tag_df['hashtag'] == tag].content.tolist()[0]
        for content in content_list:
            embed_list.append(con_emb_dict[content])
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list
    return tag_arr_dict


def rank_input_train(weibo, newTagRec, user_list, train_tag_list, user_arr_dict, tag_arr_dict, qid_train_dict, qid_embed_dict):
    if(weibo and newTagRec):
        f = open('/home/zyb/perTagRec/weibo/partData/newTagRec/bert/trainBert.dat', "a")
    elif(weibo and (not newTagRec)):
        f = open('/home/zyb/perTagRec/weibo/partData/bert/trainBert.dat', "a")
    elif((not weibo) and newTagRec):
        f = open('/home/zyb/perTagRec/twitter/newTagRec/bert/trainBert.dat', "a")
    else:
        f = open('/home/zyb/perTagRec/twitter/bert/trainBert.dat', "a")

    for user_num, user in enumerate(tqdm(user_list, desc="rank_input_train")):
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")

        if(newTagRec):
            positive_tag_list = sorted(
                list(set(qid_train_dict[user]) - set(qid_embed_dict[user])))
            temp_tag_list = sorted(
                list(set(train_tag_list) - set(positive_tag_list) - set(qid_embed_dict[user])))
        else:
            positive_tag_list = sorted(list(set(qid_train_dict[user])))
            temp_tag_list = sorted(
                list(set(train_tag_list) - set(positive_tag_list)))
        negative_tag_list = random.sample(
            temp_tag_list, 5*len(positive_tag_list))

        for tag in positive_tag_list:  # positive samples
            try:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                x = 1
                Str = f"\n{x} {'qid'}:{user_num + 1}"
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                f.write(Str)
            except:
                print(tag)

        for tag in negative_tag_list:  # negative samples
            try:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                x = 0
                Str = f"\n{x} {'qid'}:{user_num + 1}"
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                f.write(Str)
            except:
                print(tag)
        f.write("\n")


def rank_input_test(weibo, newTagRec, user_list, test_tag_list, user_arr_dict, tag_arr_dict, qid_test_dict, qid_train_dict, qid_embed_dict):
    if(weibo and newTagRec):
        f = open('/home/zyb/perTagRec/weibo/partData/newTagRec/bert/testBert.dat', "a")
        tagF = open('/home/zyb/perTagRec/weibo/partData/newTagRec/bert/tagList.txt', "a", encoding="utf-8")
    elif(weibo and (not newTagRec)):
        f = open('/home/zyb/perTagRec/weibo/partData/bert/testBert.dat', "a")
        tagF = open('/home/zyb/perTagRec/weibo/partData/bert/tagList.txt', "a", encoding="utf-8")
    elif((not weibo) and newTagRec):
        f = open('/home/zyb/perTagRec/twitter/newTagRec/bert/testBert.dat', "a")
        tagF = open('/home/zyb/perTagRec/twitter/newTagRec/bert/tagList.txt', "a", encoding="utf-8")
    else:
        f = open('/home/zyb/perTagRec/twitter/bert/testBert.dat', "a")
        tagF = open('/home/zyb/perTagRec/twitter/bert/tagList.txt',"a", encoding="utf-8")

    for user_num, user in enumerate(tqdm(user_list, desc="rank_input_test")):
        user_arr = user_arr_dict[user]
        f.write(f"# query {user_num + 1}")
        tagF.write(f"# query {user_num + 1}\n")

        if(newTagRec):
            positive_tag_list = sorted(list(
                set(qid_test_dict[user]) - set(qid_train_dict[user]) - set(qid_embed_dict[user])))
            negative_tag_list = sorted(list(set(test_tag_list) - set(
                positive_tag_list) - set(qid_train_dict[user]) - set(qid_embed_dict[user])))
        else:
            positive_tag_list = sorted(
                list(set(qid_test_dict[user]) - set(qid_train_dict[user])))
            negative_tag_list = sorted(
                list(set(test_tag_list) - set(positive_tag_list)))

        for tag in positive_tag_list:  # positive samples
            try:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                x = 1
                Str = f"\n{x} {'qid'}:{user_num + 1}"
                tagF.write(f"{tag}\n")
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                f.write(Str)
            except:
                print(tag)

        for tag in negative_tag_list:  # negative samples
            try:
                tag_arr = tag_arr_dict[tag]
                user_tag_arr = np.concatenate((user_arr, tag_arr), axis=None)
                x = 0
                Str = f"\n{x} {'qid'}:{user_num + 1}"
                tagF.write(f"{tag}\n")
                for index, value in enumerate(user_tag_arr):
                    Str += f" {index + 1}:{value}"
                f.write(Str)
            except:
                print(tag)
        f.write("\n")


def sort_user_tag(user_list, df):
    tag_list = sorted(
        list(set(train_df['hashtag'].explode('hashtag').tolist()))) #sorted(list(set())) to keep the fixed order
    qid_user_tag_dict = OrderedDict() 
    for user in tqdm(user_list, desc="sort_train_user_tag"):
        spe_user_df = df.loc[df['user_id'] == user]
        spe_user_tag_list = sorted(
            list(set(spe_user_df['hashtag'].explode('hashtag').tolist())))
        qid_user_tag_dict[user] = spe_user_tag_list
    return tag_list, qid_user_tag_dict


def read_embedding(content_df, test_df, weibo):
    #read and parse user list
    if(weibo):
        with open("/home/zyb/perTagRec/weibo/partData/userList.txt", "r", encoding="utf-8") as f:
            user_list = ast.literal_eval(f.readline().strip())[:100]
    else:
        with open("/home/zyb/perTagRec/twitter/userList.txt", "r", encoding="utf-8") as f:
            user_list = ast.literal_eval(f.readline().strip())[:150]
    
    # user1 [con1, con2, con3, ...]
    content_user_df = content_df.groupby(['user_id'], as_index=False).agg({
        'content': lambda x: list(x)})
   
    # tag1 [con1, con2, con3, ...] 
    content_tag_df = content_df.explode('hashtag').groupby(
        ['hashtag'], as_index=False).agg({'content': lambda x: list(x)})
    
    #all tags
    tag_list = sorted(
        list(set(content_df['hashtag'].explode('hashtag').tolist())))

    emb_para_list = [user_list, content_user_df, tag_list, content_tag_df]

    return emb_para_list


if __name__ == '__main__':
    '''
    python bert.py --weibo --newTagRec
    --weibo choose weibo (default: 1)
    --newTagRec run new Hashtag Recommendation (default: 1)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--weibo", default=1, type=int)
    parser.add_argument("--newTagRec", default=1, type=int)
    args = parser.parse_args()

    if(args.weibo == True):
        train_path = "/home/zyb/perTagRec/weibo/partData/train.csv"
        test_path = "/home/zyb/perTagRec/weibo/partData/test.csv"
        embed_path = "/home/zyb/perTagRec/weibo/partData/embed.csv"
        embed_json = "/home/zyb/perTagRec/weibo/partData/embed.json"
    else:
        train_path = "/home/zyb/perTagRec/twitter/train.csv"
        test_path = "/home/zyb/perTagRec/twitter/test.csv"
        embed_path = "/home/zyb/perTagRec/twitter/embed.csv"
        embed_json = "/home/zyb/perTagRec/twitter/embed.json"

    train_df = pd.read_csv(train_path, delimiter='\t', encoding="utf-8")
    test_df = pd.read_csv(test_path, delimiter='\t', encoding="utf-8")
    embed_df = pd.read_csv(embed_path, delimiter='\t', encoding="utf-8")

    #read content embedding json, get from pretrainBert.py
    with open(embed_json, 'r', encoding="utf-8") as f:
        con_emb_dict = json.load(f) 

    #format: user_id,content -->str ; hashtag -->list  
    train_df['user_id'] = train_df['user_id'].apply(str)
    train_df['content'] = train_df['content'].apply(str)
    train_df['hashtag'] = train_df['hashtag'].apply(ast.literal_eval)
    test_df['user_id'] = test_df['user_id'].apply(str)
    test_df['content'] = test_df['content'].apply(str)
    test_df['hashtag'] = test_df['hashtag'].apply(ast.literal_eval)
    embed_df['user_id'] = embed_df['user_id'].apply(str)
    embed_df['content'] = embed_df['content'].apply(str)
    embed_df['hashtag'] = embed_df['hashtag'].apply(ast.literal_eval)

    emb_para_list = read_embedding(embed_df, test_df, args.weibo)
    emb_para_list.append(con_emb_dict)

    user_arr_dict = average_user_tweet(
        emb_para_list[0], emb_para_list[1], emb_para_list[4])
    tag_arr_dict = average_hashtag_tweet(
        emb_para_list[2], emb_para_list[3], emb_para_list[4])

    train_tag, qid_train_dict = sort_user_tag(emb_para_list[0], train_df)
    test_tag, qid_test_dict = sort_user_tag(emb_para_list[0], test_df)
    if(args.weibo):
        embed_tag, qid_embed_dict = sort_user_tag(
            emb_para_list[0], embed_df[embed_df.time < 20200601])
    else:
        embed_tag, qid_embed_dict = sort_user_tag(
            emb_para_list[0], embed_df[embed_df.time < 20110201])

    rank_input_train(args.weibo, args.newTagRec, emb_para_list[0], train_tag, user_arr_dict, tag_arr_dict,
                     qid_train_dict, qid_embed_dict)
    rank_input_test(args.weibo, args.newTagRec, emb_para_list[0], test_tag, user_arr_dict, tag_arr_dict,
                    qid_test_dict, qid_train_dict, qid_embed_dict)
