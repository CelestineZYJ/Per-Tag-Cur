import numpy as np
import pandas as pd
import re
import json
import pdb
from tqdm import tqdm


def get_hashtag(content):
    """
    Get the hashtag from the content.
    """
    words = re.split(r'[:,.! "\']', str(content))
    hashtag = [word for word in words if re.search(r'^#', word)]
    # print(type(hashtag[0]))
    return hashtag


def content_embedding(content, con_emb_dict):
    try:
        return con_emb_dict[content]
    except:
        return [0]*768


def average_user_tweet(user_list, content_user_df, con_emb_dict):
    user_arr_dict = {}
    for user in user_list:
        embed_list = []
        content_list = content_user_df['content'].loc[content_user_df['user_id'] == user].tolist()[0]
        for content in content_list:
            embed_list.append(content_embedding(content, con_emb_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        user_arr_dict[user] = embed_list

    print(user_arr_dict)
    return user_arr_dict


def average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict):
    tag_arr_dict = {}
    for tag in tag_list:
        embed_list = []
        try:
            content_list = content_tag_df['content'].loc[content_tag_df['hashtag'] == tag].tolist()[0]
        except:
            pdb.set_trace()
        for content in content_list:
            embed_list.append(content_embedding(content, con_emb_dict))
        embed_list = np.mean(np.array(embed_list), axis=0)
        tag_arr_dict[tag] = embed_list

    return tag_arr_dict


def svm_rank_input(user_list, tag_list, user_arr_dict, tag_arr_dict, con_emb_dict):
    pass


if __name__ == '__main__':
    with open('./data/embeddings.json', 'r') as f:
        con_emb_dict = json.load(f)
    content_df = pd.read_table('./data/testSet.csv')
    content_df['hashtag'] = content_df['content'].apply(get_hashtag)

    user_list = list(set(content_df['user_id'].tolist()))
    content_user_df = content_df.groupby(['user_id'], as_index=False).agg({'content': lambda x: list(x)})
    tag_list = list(set(content_df.explode('hashtag')['hashtag'].tolist()))
    content_tag_df = content_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'hashtag': lambda x: list(x)})

    user_arr_dict = average_user_tweet(user_list, content_user_df, con_emb_dict)
    tag_arr_dict = average_hashtag_tweet(tag_list, content_tag_df, con_emb_dict)

    svm_rank_input(user_list, tag_list, user_arr_dict, tag_arr_dict, con_emb_dict)

