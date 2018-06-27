'''
list_one:real label
list_two:predict
'''
def cal_auc(list_one, list_two):
    '''计算AUC值'''
    positive = []
    negative = []
    for index in range(len(list_one)):
        if list_one[index] == 1:
            positive.append(index)
        else:
            negative.append(index)
    SUM = 0
    for i in positive:
        for j in negative:
            if list_two[i] > list_two[j]:
                SUM += 1
            elif list_two[i] == list_two[j]:
                SUM += 0.5
            else:
                pass
    return SUM / (len(positive)*len(negative))
