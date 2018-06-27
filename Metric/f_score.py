'''
real:真实标签 0 or 1
predict:预测结果 0 or 1
'''
def cal_f_score(real, predict):
    real = list(real)
    tp, fp, fn = 0, 0, 0
    for index in range(len(real)):
        if (real[index] == 1) and (predict[index] == 1):
            tp += 1
        elif (real[index] == 0) and (predict[index] == 1):
            fp += 1
        elif (real[index] == 1) and (predict[index] == 0):
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision:', precision)
    print('recall:', recall)
    print('f_score:', 2 * precision * recall / (precision + recall))
