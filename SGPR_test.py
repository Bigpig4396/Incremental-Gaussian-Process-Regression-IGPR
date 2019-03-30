from SGPR import SGPR
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_csv(file_name):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        columns = [row for row in reader]

    columns = np.array(columns)
    m_x, n_x = columns.shape
    data_set = np.zeros((m_x, n_x))
    for i in range(m_x):
        for j in range(n_x):
            data_set[i][j] = float(columns[i][j])
    return data_set


training_set = load_csv('training_set.csv')
training_target = load_csv('training_target.csv')
test_set = load_csv('test_set.csv')
test_target = load_csv('test_target.csv')

data_len = 500

print('iter0')
sgpr = SGPR(training_set[0, :], training_target[0, :])
print(sgpr.k_matrix)
print(sgpr.inv_k_matrix)
print(" ")
for i in range(1, data_len):
    print('iter', i)
    sgpr.learn(training_set[i, :], training_target[i, :])
    print(" ")

pred = sgpr.predict(training_set[0, :])
for i in range(1, data_len):
    pred = np.vstack((pred, sgpr.predict(training_set[i, :])))

fig = plt.figure(figsize=(5, 5))
gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[2, 1])

ax1.plot(training_target[0:data_len, 0])
ax2.plot(training_target[0:data_len, 1])
ax3.plot(training_target[0:data_len, 2])
ax4.plot(pred[0:data_len, 0])
ax5.plot(pred[0:data_len, 1])
ax6.plot(pred[0:data_len, 2])

plt.show()