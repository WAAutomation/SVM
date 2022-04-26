import numpy as np
import random
import datetime
import pickle


def load_dataset_RBF(path):
    sample = np.zeros([1, 2], dtype=float)
    sample = [sample] * 2
    temp = np.zeros([1, 2], dtype=float)
    with open(path, encoding='utf-8') as file:
        for line in file:
            line = line.strip().split()
            index = float(line[2])
            line = line[0:2]
            for i in range(2):
                temp[0][i] = float(line[i])
            if index > 0:
                sample[0] = np.append(sample[0], temp, axis=0)
            else:
                sample[1] = np.append(sample[1], temp, axis=0)
        for i in range(2):
            sample[i] = sample[i][1:len(sample[i])]
    return sample


def load_dataset_satimage(path):
    sample = np.zeros([1, 36], dtype=int)
    sample = [sample] * 7
    temp = np.zeros([1, 36], dtype=int)
    with open(path, encoding='utf-8') as file:
        for line in file:
            line = line.strip().split(', ')
            index = int(line[36])
            line = line[0:36]
            for i in range(36):
                temp[0][i] = int(line[i])
            sample[index - 1] = np.append(sample[index - 1], temp, axis=0)
        for i in range(7):
            sample[i] = sample[i][1:len(sample[i])]
        sample = [sample[0], sample[1], sample[2], sample[3], sample[4], sample[6]]
    return sample


def load_dataset(path):
    num1 = np.zeros([50, 4], dtype=float)
    num2 = np.zeros([50, 4], dtype=float)
    num3 = np.zeros([50, 4], dtype=float)
    num = np.zeros([150, 4], dtype=float)
    with open(path, encoding='utf-8') as file:
        i = 0
        for line in file:
            line = line.strip().split(',')
            for j in range(4):
                num[i][j] = float(line[j])
            i += 1
        for i in range(50):
            num1[i] = num[i]
        for i in range(50, 100):
            num2[i - 50] = num[i]
        for i in range(100, 150):
            num3[i - 100] = num[i]
        sample = [num1, num2, num3]
        return sample


def divide(dataset):
    train = []
    test = []
    for s in dataset:
        train.append(s[0:int(len(s) * 0.8)])
        test.append(s[int(len(s) * 0.8):])
    return train, test


class SVM:
    def __init__(self):
        self.feature_num = 0
        self.sample_num = 0
        self.a = 0
        self.w = 0
        self.b = 0.0
        self.E = 0
        self.C = 200
        self.inner_product = 0
        self.kernel_type = 0
        self.train_set = 0
        self.support_vector = {}
        self.tag = 0
        self.s = 1.2  # 径向基函数方差
        return

    def evaluate(self, test_set):
        test_num = test_set[0].shape[0] + test_set[1].shape[0]
        # print("测试集分类错误的样本：")
        n = 2
        TP = [0] * n
        TN = [0] * n
        FP = [0] * n
        FN = [0] * n
        for v in test_set[0]:
            # print(svm.predict(v))
            pre = self.predict(v)
            if pre <= 0:
                # print(pre, v)
                FN[0] += 1
            else:
                TP[0] += 1
        for v in test_set[1]:
            # print(svm.predict(v))
            pre = self.predict(v)
            if pre >= 0:
                # print(pre, v)
                FP[0] += 1
            else:
                TN[0] += 1
        print("测试集：")
        print("样本总数：", test_num, '\t分类错误数量：', FP[0] + FN[0])
        Accuracy = (test_num - FP[0] - FN[0]) / test_num
        Precision = TP[0] / (TP[0] + FP[0])
        Recall = TP[0] / (TP[0] + FN[0])
        F1 = 2 * Precision * Recall / (Recall + Precision)
        print("正确率\t准确率\t召回率\tF1值\t")
        print(format(Accuracy, '.3f'), '\t', format(Precision, '.3f'), '\t', format(Recall, '.3f'), '\t',
              format(F1, '.3f'), '\t')
        # print("训练集分类错误的样本：")
        for i in range(self.sample_num):
            pre = self.predict(self.train_set[i])
            if self.tag[i] > 0:
                if pre <= 0:
                    FN[1] += 1
                else:
                    TP[1] += 1
            else:
                if pre >= 0:
                    FP[1] += 1
                else:
                    TN[1] += 1
        print("训练集：")
        print("样本总数：", self.sample_num, '\t分类错误数量：', FP[1] + FN[1])
        Accuracy = (self.sample_num - FP[1] - FN[1]) / self.sample_num
        Precision = TP[1] / (TP[1] + FP[1])
        Recall = TP[1] / (TP[1] + FN[1])
        F1 = 2 * Precision * Recall / (Recall + Precision)
        print("正确率\t准确率\t召回率\tF1值\t")
        print(format(Accuracy, '.3f'), '\t', format(Precision, '.3f'), '\t', format(Recall, '.3f'), '\t',
              format(F1, '.3f'), '\t')

    def load(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def save(self, path):
        file = open(path, 'w').close()
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def kernel_function(self, vector1, vector2):
        if self.kernel_type == 0:
            return np.dot(vector1, vector2)
        elif self.kernel_type == 1:
            return np.exp(
                np.sum((vector1 - vector2) * (vector1 - vector2)) / (-1 * self.s ** 2))
        else:
            raise NameError('核函数无法识别！')

    def predict(self, vector):
        if vector.shape != self.w.shape:
            raise NameError('输入向量与训练集维度不一致！')
        result = self.b
        for i in self.support_vector.keys():
            result += self.a[i] * self.tag[i] * self.kernel_function(self.train_set[i], vector)
        return result

    def select(self, i, Ei):
        # 使用启发式规则优化a[j]的选择
        res = -1
        E_max = -1
        for j in self.support_vector.keys():
            if i != j:
                Ej = self.E[j].copy()
                if abs(Ei - Ej) > E_max:
                    E_max = abs(Ei - Ej)
                    res = j
        if E_max == -1:
            res = random.randint(0, self.sample_num - 1)
            while res == i:
                res = random.randint(0, self.sample_num - 1)
        return res, self.E[res].copy()

    def update_ai(self, i):
        temp = self.E[i] * self.tag[i]
        # 判断是否违反KKT条件
        if (temp < -0.0001 and self.a[i] < self.C) or (
                temp > 0.0001 and self.a[i] > 0):
            # 如果违反
            # 选择|Ei-Ej|最大的作为j
            Ei = self.E[i].copy()
            j, Ej = self.select(i, Ei)
            # 计算边界
            if self.tag[i] != self.tag[j]:
                L = max(0, self.a[j] - self.a[i])
                H = min(self.C, self.C + self.a[j] - self.a[i])
            else:
                L = max(0, self.a[j] + self.a[i] - self.C)
                H = min(self.C, self.a[i] + self.a[j])
            if L == H:
                return 0
            # 计算学习率
            n = 2 * self.inner_product[i][j] - self.inner_product[i][i] - self.inner_product[j][j]
            if n >= 0:
                return 0
            # 更新ai,aj
            aj_old = self.a[j].copy()
            ai_old = self.a[i].copy()
            self.a[j] -= (self.tag[j] * (Ei - Ej)) / n
            # 判断边界
            if self.a[j] > H:
                self.a[j] = H
            if self.a[j] < L:
                self.a[j] = L
            # 维护support_vector
            if 0 < self.a[j]:
                self.support_vector[j] = self.a[j]
            elif j in self.support_vector:
                self.support_vector.pop(j)
            if abs(aj_old - self.a[j]) < 0.00001:
                self.E = self.E - aj_old * self.tag[j] * self.inner_product[j]
                self.E = self.E + self.a[j] * self.tag[j] * self.inner_product[j]
                return 0
            self.a[i] += self.tag[i] * self.tag[j] * (aj_old - self.a[j])
            # 维护support_vector
            if 0 < self.a[i]:
                self.support_vector[i] = self.a[i]
            elif i in self.support_vector:
                self.support_vector.pop(i)
            # 更新b
            b_old = self.b
            b1 = self.b - Ei - self.tag[i] * (self.a[i] - ai_old) * self.inner_product[i][i] - self.tag[j] * (
                    self.a[j] - aj_old) * self.inner_product[j][i]
            b2 = self.b - Ej - self.tag[i] * (self.a[i] - ai_old) * self.inner_product[i][j] - self.tag[j] * (
                    self.a[j] - aj_old) * self.inner_product[j][j]
            if 0 < self.a[i] < self.C:
                self.b = b1
            elif 0 < self.a[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            # 更新E
            self.E = self.E - b_old - ai_old * self.tag[i] * self.inner_product[i]
            self.E = self.E + self.b + self.a[i] * self.tag[i] * self.inner_product[i]
            self.E = self.E - aj_old * self.tag[j] * self.inner_product[j]
            self.E = self.E + self.a[j] * self.tag[j] * self.inner_product[j]
            return 1
        return 0

    def train_SMO(self, type1, type2, C=float('inf'), limit=100, kernel_type=0, s=1.2):
        self.s = s
        self.C = C
        self.kernel_type = kernel_type
        shape1 = type1.shape
        shape2 = type2.shape
        if shape1[1] != shape2[1]:
            print("训练集特征数不相同！")
            return
        # 根据训练集初始化参数
        self.sample_num = shape1[0] + shape2[0]
        self.feature_num = shape1[1]
        temp = np.zeros(shape2[0], dtype='float64')
        temp -= 1.0
        self.tag = np.concatenate((np.ones(shape1[0]), temp), dtype='float64')
        self.E = -self.tag
        self.train_set = np.concatenate((type1, type2), dtype='float64')
        self.inner_product = np.zeros((self.sample_num, self.sample_num), dtype='float64')
        for i in range(self.sample_num):
            for j in range(self.sample_num):
                self.inner_product[i][j] = self.kernel_function(self.train_set[i], self.train_set[j])
        self.a = np.zeros(self.sample_num, dtype='float64')  # 初始化α,w,b
        self.w = np.zeros(self.feature_num)
        self.b = 0.0
        ischange = 1
        count = 0
        while ischange > 0 and count < limit:
            count += 1
            ischange = 0
            for i in range(self.sample_num):
                ischange += self.update_ai(i)
        self.w = np.zeros(self.feature_num)
        for i in range(self.sample_num):
            self.w += self.a[i] * self.tag[i] * self.train_set[i]
        return


if __name__ == "__main__":
    svm = SVM()
    # random.seed(10)
    sample = load_dataset("iris.data")  # C = 1.0, s = 1.5
    # sample = load_dataset_RBF("testSetRBF.txt")
    # sample = load_dataset_satimage("satimage.dat")  # C = 150, s = 50
    train, test = divide(sample)
    # svm = svm.load("model.txt")
    starttime = datetime.datetime.now()
    m = 0
    n = 1
    svm.train_SMO(train[m], train[n], C=200, limit=100, kernel_type=1, s=1.5)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    svm.evaluate(test[m:m + 1] + test[n:n + 1])
    svm.save("model.txt")
