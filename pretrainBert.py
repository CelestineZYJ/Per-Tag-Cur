import torch
from transformers import BertModel, BertTokenizer #AutoModel, AutoTokenizer
import pandas as pd
import json
import numpy as np


def get_str(content):
    Str = str(content)
    return Str


def cal_bert(embed_df):
    embed_df['user_id'] = embed_df['user_id'].apply(get_str)
    embed_df['content'] = embed_df['content'].apply(get_str)
    lines = embed_df['content'].tolist()

    bertTweet = BertModel.from_pretrained("hfl/chinese-bert-wwm")  # “vinai/bertweet-base”
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")  # “vinai/bertweet-base”
    # INPUT TWEET IS ALREADY NORMALIZED!

    con_emb_dict = {}
    for index, line in enumerate(lines):
        print(str(index)+'  '+line)
        input_ids = torch.tensor([tokenizer.encode(line)])
        if len(input_ids[0].numpy().tolist()) > 128:
            input_ids = torch.from_numpy(np.array(input_ids[0].numpy().tolist()[0:128])).reshape(1, -1).type(torch.LongTensor)

        with torch.no_grad():
            features = bertTweet(input_ids)  # Models outputs are now tuples
        con_emb_dict[line] = features[1].numpy()[0].tolist()

    jsObj = json.dumps(con_emb_dict)

    fileObj = open('./wData/embeddings.json', 'w')
    fileObj.write(jsObj)
    fileObj.close()


def test_dict(embed_df, con_emb_dict):
    print(con_emb_dict)
    content_list = embed_df['content'].tolist()
    for content in content_list:
        try:
            print(con_emb_dict[content])
        except:
            print(content)
            print('\n')


if __name__ == '__main__':
    embed_df = pd.read_table('./wData/embed.csv')
    #embed_df = embed_df[:200]
    cal_bert(embed_df)
'''
    with open('./weiData/embeddings.json', 'r') as f:
        con_emb_dict = json.load(f)
    test_dict(embed_df, con_emb_dict)
'''

