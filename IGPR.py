from time import sleep

import numpy as np
import csv
from collections import deque
import torch
import random

from sklearn.metrics import accuracy_score


class HyperParam(object):
    def __init__(self, theta_f=1, len=1, theta_n=0.1):
        self.theta_f = theta_f  # for squared exponential kernel
        self.len = len  # for squared exponential kernel
        self.theta_n = theta_n  # for squared exponential kernel


class IGPR(object):
    def __init__(self, init_x, init_y, maxSize=50, device=None):
        # init_x should be an array shape(j,)
        # init_y should be an array shape(m,)
        self.hyperparam = HyperParam(1, 1, 0.1)
        self.device = device
        self.max_k_matrix_size = maxSize
        self.lamda = 1
        self.count = 0
        self.kernel_x = torch.tensor(init_x, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.kernel_y = torch.tensor(init_y, dtype=torch.float32, device=self.device).reshape(1, -1)
        self.k_matrix = torch.ones(1, dtype=torch.float32,
                                   device=self.device) + self.hyperparam.theta_n * self.hyperparam.theta_n
        self.inv_k_matrix = torch.ones(1, dtype=torch.float32, device=self.device) / (
                self.hyperparam.theta_n * self.hyperparam.theta_n)
        self.is_av = False
        self.delta = torch.sum(self.k_matrix, dim=0)

    def is_available(self):
        n = len(self.kernel_x)
        if n >= 2:
            self.is_av = True
        return self.is_av

    def learn(self, new_x, new_y):
        # new_x should be an array shape(j,)
        # new_y should be an array shape(m,)
        new_x = torch.tensor(new_x, dtype=torch.float32, device=self.device)
        new_y = torch.tensor(new_y, dtype=torch.float32, device=self.device)
        self.delta = self.delta * self.lamda

        if self.is_available():
            if len(self.kernel_x) < self.max_k_matrix_size:
                self.aug_update_SE_kernel(new_x, new_y)
            else:
                new_delta = self.count_delta(new_x)
                max_value, max_index = self.get_max(self.delta)
                if new_delta < max_value:
                    # self.schur_update_SE_kernel(new_x, new_y)
                    # print('SM_update_SE_kernel')
                    self.SM_update_SE_kernel(new_x, new_y, max_index)
                    self.count = self.count + 1
                    if self.count > 100:
                        self.count = 0
                        self.calculate_SE_kernel()
                        self.inv_k_matrix = torch.inverse(self.k_matrix)


        else:
            self.kernel_x = torch.cat((self.kernel_x, new_x.reshape(1, -1)), 0)
            self.kernel_y = torch.cat((self.kernel_y, new_y.reshape(1, -1)), 0)
            self.calculate_SE_kernel()
            self.inv_k_matrix = torch.inverse(self.k_matrix)

    def calculate_SE_kernel(self):
        n = len(self.kernel_x)
        self.k_matrix = torch.zeros((n, n), dtype=torch.float32, device=self.device)

        for index in range(n):
            kernel_index = self.kernel_x[index].repeat(n, 1)
            a = -2 * self.hyperparam.len * self.hyperparam.len
            b = self.hyperparam.theta_f * self.hyperparam.theta_f
            self.k_matrix[:, index] = torch.exp(torch.sum(torch.square(self.kernel_x - kernel_index), 1) / a) * b

        self.k_matrix = self.k_matrix + self.hyperparam.theta_n * self.hyperparam.theta_n * torch.eye(n,
                                                                                                      dtype=torch.float32,
                                                                                                      device=self.device)

        self.delta = torch.sum(self.k_matrix, 0)

    def predict(self, coming_x):
        coming_x = torch.tensor(coming_x, dtype=torch.float32, device=self.device)
        if self.is_available():
            n = len(self.kernel_x)
            new_x_square = coming_x.repeat(n, 1)
            a = -2 * self.hyperparam.len * self.hyperparam.len
            b = self.hyperparam.theta_f * self.hyperparam.theta_f
            cross_kernel_k = torch.exp(torch.sum(torch.square(self.kernel_x - new_x_square), 1) / a) * b

            prediction = cross_kernel_k.dot(self.inv_k_matrix.mm(self.kernel_y.reshape(-1, 1)).reshape(-1, ))
        else:
            prediction = self.kernel_y[0]
        return prediction.cpu().item()

    def aug_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)
        self.kernel_x = torch.cat((self.kernel_x, new_x.reshape(1, -1)), 0)
        self.kernel_y = torch.cat((self.kernel_y, new_y.reshape(1, -1)), 0)
        # add the extra column an row for the new data
        temp = torch.zeros(n + 1, n + 1, device=self.device)
        temp[:n, :n] = self.k_matrix
        self.k_matrix = temp

        # calculate the new column of data
        new_x_square = new_x.repeat(n + 1, 1)
        a = -2 * self.hyperparam.len * self.hyperparam.len
        b = self.hyperparam.theta_f * self.hyperparam.theta_f
        self.k_matrix[:, n] = torch.exp(torch.sum(torch.square(self.kernel_x - new_x_square), 1) / a) * b

        # update the last point (new point) in the diagonal
        self.k_matrix[n, n] = self.k_matrix[n, n] + self.hyperparam.theta_n * self.hyperparam.theta_n

        # copy the column of data to the row (they're the same values)
        self.k_matrix[n, 0:n] = self.k_matrix[0:n, n]

        b = self.k_matrix[0:n, n]  # shape(n,)
        d = self.k_matrix[n, n]  # scalar
        # e = self.inv_k_matrix.dot(b)
        e = torch.sum(self.inv_k_matrix * b, 1)  # shape(n,)
        # print('e', e)
        # g = 1 / (d - (b.T).dot(e))
        g = 1 / (d - torch.dot(b, e))  # scalar
        # print('g', g)
        haha_11 = self.inv_k_matrix + g * e * e.T
        haha_12 = -g * e.reshape(-1, 1)
        haha_21 = haha_12.T
        haha_22 = torch.tensor([[g]], dtype=torch.float32, device=self.device)

        temp_1 = torch.cat((haha_11, haha_12), 1)
        temp_2 = torch.cat((haha_21, haha_22), 1)
        self.inv_k_matrix = torch.cat((temp_1, temp_2), 0)

        # update delta
        self.delta = self.delta + self.k_matrix[:n, n]
        self.delta = torch.cat((self.delta, torch.tensor([0], dtype=torch.float32, device=self.device)))
        self.delta[n] = torch.sum(self.k_matrix[:, n])

    def schur_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)

        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        self.kernel_x.popleft()
        self.kernel_y.popleft()

        K2 = np.zeros((n, n))
        K2[0:n - 1, 0:n - 1] = self.k_matrix[1:n, 1:n]
        for i in range(n):
            K2[i, n - 1] = np.sum(np.square(self.kernel_x[i] - new_x))
            K2[i, n - 1] = K2[i, n - 1] / (-2)
            K2[i, n - 1] = K2[i, n - 1] / self.hyperparam.len
            K2[i, n - 1] = K2[i, n - 1] / self.hyperparam.len
            K2[i, n - 1] = np.exp(K2[i, n - 1])
            K2[i, n - 1] = K2[i, n - 1] * self.hyperparam.theta_f
            K2[i, n - 1] = K2[i, n - 1] * self.hyperparam.theta_f

        K2[n - 1, n - 1] = K2[n - 1, n - 1] + self.hyperparam.theta_n * self.hyperparam.theta_n
        K2[n - 1, 0:n - 1] = (K2[0:n - 1, n - 1]).T

        # print('k_matrix', self.k_matrix)
        # print('new k_matrix', K2)
        # print('inv_k_matrix', self.inv_k_matrix)
        e = self.inv_k_matrix[0][0]
        # print('e', e)
        f = self.inv_k_matrix[1:n, 0].reshape((n - 1, 1))
        # print('f', f)
        g = K2[n - 1, n - 1]
        # print('g', g)
        h = K2[0:n - 1, n - 1].reshape((n - 1, 1))
        # print('h', h)
        H = self.inv_k_matrix[1:n, 1:n]
        # print('H', H)
        B = H - (f.dot(f.T)) / e
        # print('B', B)
        s = 1 / (g - (h.T).dot(B.dot(h)))
        # print('s', s)
        haha_11 = B + (B.dot(h)).dot((B.dot(h)).T) * s
        haha_12 = -B.dot(h) * s
        haha_21 = -(B.dot(h)).T * s
        haha_22 = s
        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        # update delta
        self.delta.popleft()
        self.delta.append(0)
        for i in range(n - 1):
            self.delta[i] = self.delta[i] - self.k_matrix[0, i + 1]

        for i in range(n - 1):
            self.delta[i] = self.delta[i] + K2[n - 1, i]

        for i in range(n):
            self.delta[n - 1] = self.delta[n - 1] + K2[i, n - 1]

        self.k_matrix = K2

    def SM_update_SE_kernel(self, new_x, new_y, index):
        n = len(self.kernel_x)
        self.kernel_x[index] = new_x
        self.kernel_y[index] = new_y
        new_k_matrix = self.k_matrix.clone()

        new_x_square = new_x.repeat(n, 1)
        a = -2 * self.hyperparam.len * self.hyperparam.len
        b = self.hyperparam.theta_f * self.hyperparam.theta_f
        new_k_matrix[:, index] = torch.exp(torch.sum(torch.square(self.kernel_x - new_x_square), 1) / a) * b

        new_k_matrix[index, index] = new_k_matrix[index, index] + self.hyperparam.theta_n * self.hyperparam.theta_n

        # copy the column of data to the row
        new_k_matrix[index, :] = new_k_matrix[:, index]

        r = (new_k_matrix[:, index] - self.k_matrix[:, index]).reshape(-1, 1)

        A = self.inv_k_matrix - \
            torch.mm(self.inv_k_matrix, r * self.inv_k_matrix[index, :].reshape(1, -1)) / \
            (1 + torch.dot(r.reshape(-1, ), self.inv_k_matrix[:, index]))

        self.inv_k_matrix = A - \
                            A[:, index].reshape(-1, 1) * torch.sum(r * A, 0) / \
                            (1 + r.reshape(-1, ).dot(A[:, index]))

        # update delta
        self.delta = self.delta - self.k_matrix[index, :] + new_k_matrix[index, :]
        self.delta[index] = torch.sum(new_k_matrix[:, index])

        self.k_matrix = new_k_matrix

    def count_delta(self, new_x):
        n = len(self.kernel_x)
        new_x_square = new_x.repeat(n, 1)
        a = -2 * self.hyperparam.len * self.hyperparam.len
        b = self.hyperparam.theta_f * self.hyperparam.theta_f
        tmp = torch.exp(torch.sum(torch.square(self.kernel_x - new_x_square), 1) / a) * b

        return torch.sum(tmp).item()

    def get_max(self, delta):
        max_index = torch.argmax(delta).item()
        max_value = delta[max_index].item()
        return max_value, max_index


if __name__ == '__main__':
    XTrain = np.arange(4 * 1000).reshape(-1, 4)
    y = list(range(1000))
    random.Random(4).shuffle(y)
    YTrain = np.array(y).reshape(-1, 1)

    m = IGPR(XTrain[0], YTrain[0], maxSize=20, device=torch.device("cuda"))

    for i in range(1, len(XTrain)):
        m.learn(XTrain[i], YTrain[i])

    pred = []
    for i in range(len(XTrain)):
        pred.append(m.predict(XTrain[i]))
    pred = np.array(pred)
    print(accuracy_score(YTrain.reshape(-1), pred.reshape(-1) > .5))
