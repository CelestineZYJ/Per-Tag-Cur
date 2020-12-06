import pandas as pd
from tqdm import tqdm
from operator import itemgetter
from tqdm import tqdm

def sort(test,predict_scores,sorted_test_path):
    test = test.readlines()
    scores = predict_scores.readlines()

    data = dict()
    key = test[0]
    value = []

    for line in tqdm(test[1:],desc='Read original test'):
        if(line[0] == '#'):
            data[key] = value
            key = line
            value = [] 
        else:
            value.append(line)

    count = 0
    for user, tags in tqdm(data.items(),desc='Write sorted test'):
        temp_scores = scores[count:count+len(tags)]
        sort_tags,sort_scores = [list(x) for x in zip(*sorted(zip(tags, temp_scores), key=itemgetter(1), reverse=True))]

        with open("sorted_test_path","a",encoding="utf-8") as sorted_test:
             sorted_test.write(user)
             sorted_test.writelines(sort_tags)
        count += len(tags)
    
def main():
    testLda = open("/Users/zyb/Desktop/NLP/Person-Hash-Predict/testLda.dat/testLda.dat","r",encoding="utf-8")
    preLdaSvm = open("/Users/zyb/Desktop/NLP/Person-Hash-Predict/testLda.dat/preLdaSvm.txt","r",encoding="utf-8")
    sorted_test_path = "/Users/zyb/Desktop/NLP/Person-Hash-Predict/testLda.dat/sortedTestLda.txt"
    sort(testLda,preLdaSvm,sorted_test_path)

if __name__ == "__main__":
    main()