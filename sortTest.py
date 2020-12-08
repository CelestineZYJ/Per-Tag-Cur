def getPre(elem):
    return elem[1]


def sortTest(testDat, predict, sort):
    tests = testDat.readlines()
    pres = predict.readlines()
    f = open(sort, "a")

    i = -1  #
    j = -1  #
    spe_user_list = []
    for line in tests:
        i += 1
        j += 1
        if line[0] == '#':
            spe_user_list.sort(key=getPre, reverse=True)
            print(spe_user_list)
            str = ""
            for lis in spe_user_list:
                str += lis[0]
            f.write(str)
            j -= 1
            spe_user_list = []
            continue
        if i == len(tests)-1:
            tests[i] = (tests[i], int(pres[j]))
            spe_user_list.append(tests[i])
            spe_user_list.sort(key=getPre, reverse=True)
            print(spe_user_list)
            str = ""
            for lis in spe_user_list:
                str += lis[0]
            f.write(str)
            j -= 1
            spe_user_list = []
        tests[i] = (tests[i], int(pres[j]))
        spe_user_list.append(tests[i])


if __name__ == "__main__":
    testDat = open("tLda/testLda.dat")
    predict = open("tLda/preLdaSvm.txt")
    sort = "tLda/sortLdaSvm.txt"
    sortTest(testDat, predict, sort)
