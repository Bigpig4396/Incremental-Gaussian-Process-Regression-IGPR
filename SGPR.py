
import numpy as np
import csv
from collections import deque

class HyperParam(object):
    def __init__(self, theta_f=1, len=1, theta_n=0.1):
        self.theta_f = theta_f       # for squared exponential kernel
        self.len = len           # for squared exponential kernel
        self.theta_n = theta_n     # for squared exponential kernel

class SGPR(object):
    def __init__(self, init_x, init_y):
        self.hyperparam = HyperParam(1, 1, 0.1)
        self.max_k_matrix_size = 500
        self.kernel_x = deque(maxlen=self.max_k_matrix_size)
        self.kernel_y = deque(maxlen=self.max_k_matrix_size)
        self.kernel_x.append(init_x)
        self.kernel_y.append(init_y)
        self.k_matrix = np.ones((1, 1)) + self.hyperparam.theta_n * self.hyperparam.theta_n
        self.inv_k_matrix = np.ones((1, 1)) / (self.hyperparam.theta_n * self.hyperparam.theta_n)
        self.is_av = False

    def is_available(self):
        n = len(self.kernel_x)
        if n >= 2:
            self.is_av = True
        return self.is_av

    def load_csv(self, file_name):
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            columns = [row for row in reader]

        columns = np.array(columns)
        m_x, n_x = columns.shape
        data_set = np.zeros((m_x,n_x))
        for i in range(m_x):
            for j in range(n_x):
                data_set[i][j] = float(columns[i][j])
        return data_set

    def learn(self, new_x, new_y):
        if self.is_available():
            print('available')
            if len(self.kernel_x) < self.max_k_matrix_size:
                self.aug_update_SE_kernel(new_x, new_y)
            else:
                self.schur_update_SE_kernel(new_x, new_y)
        else:
            self.kernel_x.append(new_x)
            self.kernel_y.append(new_y)
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)

    def calculate_SE_kernel(self):
        n = len(self.kernel_x)
        self.k_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.k_matrix[i][j] = np.sum(np.square(self.kernel_x[i] - self.kernel_x[j]))
                self.k_matrix[i][j] = self.k_matrix[i][j] / (-2)
                self.k_matrix[i][j] = self.k_matrix[i][j] / self.hyperparam.len
                self.k_matrix[i][j] = self.k_matrix[i][j] / self.hyperparam.len
                self.k_matrix[i][j] = np.exp(self.k_matrix[i][j])
                self.k_matrix[i][j] = self.k_matrix[i][j] * self.hyperparam.theta_f
                self.k_matrix[i][j] = self.k_matrix[i][j] * self.hyperparam.theta_f
        self.k_matrix = self.k_matrix + self.hyperparam.theta_n * self.hyperparam.theta_n * np.eye(n)

    def predict(self, coming_x):
        if self.is_available():
            n = len(self.kernel_x)
            cross_kernel_k = np.zeros((1, n))
            for i in range(n):
                cross_kernel_k[0, i] = np.sum(np.square(self.kernel_x[i] - coming_x))
                cross_kernel_k[0, i] = cross_kernel_k[0, i] / (-2)
                cross_kernel_k[0, i] = cross_kernel_k[0, i] / self.hyperparam.len
                cross_kernel_k[0, i] = cross_kernel_k[0, i] / self.hyperparam.len
                cross_kernel_k[0, i] = np.exp(cross_kernel_k[0, i])
                cross_kernel_k[0, i] = cross_kernel_k[0, i] * self.hyperparam.theta_f
                cross_kernel_k[0, i] = cross_kernel_k[0, i] * self.hyperparam.theta_f
            kernel_y_mat = self.kernel_y[0]
            for i in range(1, n):
                kernel_y_mat = np.vstack((kernel_y_mat, self.kernel_y[i]))
            # print('kernel_y',self.kernel_y)
            # print('kernel_y_mat', kernel_y_mat)
            prediction = cross_kernel_k.dot(self.inv_k_matrix.dot(kernel_y_mat))
        else:
            prediction = self.kernel_y[0]
        return prediction

    def aug_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)
        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        self.k_matrix = np.hstack((self.k_matrix, np.zeros((n, 1))))
        self.k_matrix = np.vstack((self.k_matrix, np.zeros((1, n+1))))

        for i in range(n+1):
            self.k_matrix[i, n] = np.sum(np.square(self.kernel_x[i] - new_x))
            self.k_matrix[i, n] = self.k_matrix[i, n] / (-2)
            self.k_matrix[i, n] = self.k_matrix[i, n] / self.hyperparam.len
            self.k_matrix[i, n] = self.k_matrix[i, n] / self.hyperparam.len
            self.k_matrix[i, n] = np.exp(self.k_matrix[i, n])
            self.k_matrix[i, n] = self.k_matrix[i, n] * self.hyperparam.theta_f
            self.k_matrix[i, n] = self.k_matrix[i, n] * self.hyperparam.theta_f

        self.k_matrix[n, n] = self.k_matrix[n, n] + self.hyperparam.theta_n * self.hyperparam.theta_n
        self.k_matrix[n, 0:n] = (self.k_matrix[0:n, n]).T
        b = self.k_matrix[0:n, n].reshape((n, 1))
        # print('b', b)
        d = self.k_matrix[n, n]
        # print('d', d)
        e = self.inv_k_matrix.dot(b)
        # print('e', e)
        g = 1 / (d - (b.T).dot(e))
        # print('g', g)
        haha_11 = self.inv_k_matrix + g[0][0]*e.dot(e.T)
        haha_12 = -g[0][0]*e
        haha_21 = -g[0][0]*(e.T)
        haha_22 = g


        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

    def schur_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)

        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)

        K2 = np.zeros((n, n))
        K2[0:n-1, 0:n-1] = self.k_matrix[1:n, 1:n]
        for i in range(n):
            K2[i, n-1] = np.sum(np.square(self.kernel_x[i] - new_x))
            K2[i, n-1] = K2[i, n-1] / (-2)
            K2[i, n-1] = K2[i, n-1] / self.hyperparam.len
            K2[i, n-1] = K2[i, n-1] / self.hyperparam.len
            K2[i, n-1] = np.exp(K2[i, n-1])
            K2[i, n-1] = K2[i, n-1] * self.hyperparam.theta_f
            K2[i, n-1] = K2[i, n-1] * self.hyperparam.theta_f

        K2[n-1, n-1] = K2[n-1, n-1] + self.hyperparam.theta_n * self.hyperparam.theta_n
        K2[n-1, 0:n-1] = (K2[0:n-1, n-1]).T

        # print('k_matrix', self.k_matrix)
        # print('new k_matrix', K2)
        # print('inv_k_matrix', self.inv_k_matrix)
        e = self.inv_k_matrix[0][0]
        # print('e', e)
        f = self.inv_k_matrix[1:n, 0].reshape((n-1, 1))
        # print('f', f)
        g = K2[n-1, n-1]
        # print('g', g)
        h = K2[0:n-1, n-1].reshape((n-1, 1))
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
        self.k_matrix = K2
