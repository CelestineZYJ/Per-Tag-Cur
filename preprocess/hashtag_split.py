import os
from wordseg import segment

hashtag_set = set()
for file in ['train.csv', 'valid.csv', 'test.csv']:
    ab_file = os.path.join('/Users/ziyjiang/Desktop/', file)
    with open(ab_file) as f:
        for line in f:
            l = line.strip('\n').split('\t')
            for hashtag in l[2:]:
                if len(hashtag) > 0:
                    hashtag_set.add(hashtag)

f = open('hashtag_split_trec_update.csv', 'w')
for hashtag in hashtag_set:
    f.write(hashtag)
    f.write('\t')
    if len(hashtag) >= 10:
        hashtag_split = []
        tmp, _ = segment(hashtag)
        for tt in tmp:
            if len(tt) > 2 and tt != 'the':
                hashtag_split.append(tt)
        if len(hashtag_split) == 0:
            hashtag_split = [hashtag]

    else:
        hashtag_split = [hashtag]
    for tt in hashtag_split:
        f.write(tt)
        f.write('\t')
    f.write('\n')
