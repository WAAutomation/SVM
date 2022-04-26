import random
from SVM import *


class SVM_n:
    def __init__(self):
        self.n = 0
        self.svm_list = []
        return

    def eva(self, test):
        n = len(test)
        test_num = 0
        TP = [0] * n
        FP = [0] * n
        FN = [0] * n
        Precision = [0] * n
        Recall = [0] * n
        for i in range(n):
            test_num += test[i].shape[0]
            for v in test[i]:
                pre = self.predict(v)
                if pre != i:
                    # print(pre, i, v)
                    FN[i] += 1
                    FP[pre] += 1
                else:
                    TP[i] += 1
        temp = 0
        for i in range(n):
            temp += FN[i]
            Precision[i] = TP[i] / (TP[i] + FP[i])
            Recall[i] = TP[i] / (TP[i] + FN[i])
        precision = 0
        recall = 0
        for i in range(n):
            precision += Precision[i] * test[i].shape[0] / test_num
            recall += Recall[i] * test[i].shape[0] / test_num
        print("样本总数：", test_num, '\t分类错误数量：', temp)
        Accuracy = (test_num - temp) / test_num
        F1 = 2 * precision * recall / (recall + precision)
        print("正确率\t准确率\t召回率\tF1值\t")
        print(format(Accuracy, '.3f'), '\t', format(precision, '.3f'), '\t', format(recall, '.3f'), '\t',
              format(F1, '.3f'), '\t')

    def evaluate(self, train, test):
        print("测试集：")
        self.eva(test)
        print("训练集：")
        self.eva(train)

    def load(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def save(self, path):
        file = open(path, 'w').close()
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def train(self, sample, C=1.0, limit=100, kernel_type=0, s=1.5):
        self.n = len(sample)
        # 在每两个类别之间训练一个分类器
        for i in range(0, self.n - 1):
            temp_list = []
            for j in range(i + 1, self.n):
                temp = SVM()
                temp.train_SMO(sample[i], sample[j], C=C, limit=limit, kernel_type=kernel_type, s=s)
                temp_list.append(temp)
            self.svm_list.append(temp_list)

    def predict(self, vector):
        vote = [0] * self.n
        probability_sum = [0] * self.n
        for i in range(0, self.n - 1):
            for j in range(i + 1, self.n):
                pre = self.svm_list[i][j - i - 1].predict(vector)
                # print(pre)
                if pre > 0:
                    vote[i] += 1
                    probability_sum[i] += pre
                elif pre < 0:
                    vote[j] += 1
                    probability_sum[j] += pre
        result = 0
        max_pro = -1
        max_vote = 0
        for i in range(self.n):
            if vote[i] > max_vote:
                result = i
                max_pro = probability_sum[i]
                max_vote = vote[i]
            elif vote[i] == max_vote and probability_sum[i] > max_pro:
                result = i
                max_pro = probability_sum[i]
                max_vote = vote[i]
        return result


if __name__ == "__main__":
    svm_n = SVM_n()
    random.seed(3)
    # svm_n = svm_n.load("model_n_150_50.txt")

    sample = load_dataset("iris.data")  # C = 200, s = 1.5
    # sample = load_dataset_RBF("testSetRBF.txt")
    # sample = load_dataset_satimage("satimage.dat")  # C = 150, s = 50
    train, test = divide(sample)

    starttime = datetime.datetime.now()
    svm_n.train(train, C=200, limit=100, kernel_type=1, s=1.5)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    # svm_n.save("model_n.txt")

    svm_n.evaluate(train, test)
