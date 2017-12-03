'''计算F1值-宏平均'''
'''test_data is dataframe'''
def F_score(test_data):
    test_data.columns = ['0', '1']
    list_one = list(test_data['0'])
    list_two = list(test_data['1'])
    print(list_one)
    print(list_two)
    sum_f = 0
    for index in range(test_data.shape[0]):
        word_one = set(list_one[index].split(','))
        word_two = list_two[index].split(',')
        p = 0; r = 0; sum = 0
        for string in word_two:
            if string in word_one:
                sum += 1
        p = sum / len(word_one)
        r = sum / len(word_two)
        # print(sum, len(word_one), len(word_two))
        if p != 0 or r != 0:
            sum_f += (2 * p * r) / (p + r)
    print('F-score:', sum_f / len(test_data))
