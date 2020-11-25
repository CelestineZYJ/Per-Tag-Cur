import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import json

embed_df = pd.read_table('./data/testSet1.csv')
embed_df = embed_df['content']
print(embed_df.loc[897])

lines = embed_df.tolist()

bertTweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
# INPUT TWEET IS ALREADY NORMALIZED!

con_emb_dict = {}
for index, line in enumerate(lines):
    print(str(index)+line)
    input_ids = torch.tensor([tokenizer.encode(line)])

    with torch.no_grad():
        features = bertTweet(input_ids)  # Models outputs are now tuples
    con_emb_dict[line] = features

print(con_emb_dict)
'''
jsObj = json.dumps(con_emb_dict)

fileObj = open('embeddings.json', 'w')
fileObj.write(jsObj)
fileObj.close()
 
## with Tensorflow 2.0+:
# from transformers import TFAutoModel
# bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
'''