import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

def get_u():  # input  range[0,0.5)
    return np.random.random() * 0.5

# s(t+1) = 0.3s(t)+0.05s(t)∑(9,i=0) s(t−i)+1.5u(t−9)u(t)+0.1
# s(t+1) = tanh(0.3s(t) + 0.05s(t)∑(19,i=0) s(t − i) +1.5u(t − 19)u(t) + 0.01) + 0.2,
# s(t+1) = 0.2s(t)+0.004s(t)∑(29,i=0)s(t−i)+1.5u(t−29)u(t)+0.201
u = []
s = [0]
for t in range(1000):
    u.append(get_u())
    tmps = 0.3 * s[t] + 0.05 * s[t] * (sum(s[-10:]) if len(s) >= 10 else sum(s)) + (
        1.5 * u[t - 10] * u[t] if len(u) >= 10 else 0) + 0.1
    s.append(tmps)

for t in range(1000, 2000):
    u.append(get_u())
    tmps = np.tanh(0.3 * s[t] + 0.05 * s[t] * sum(s[-20:]) + 1.5 * u[-20] * u[t] + 0.01) + 0.2
    s.append(tmps)

for t in range(2000, 3001):
    u.append(get_u())
    tmps = 0.2 * s[t] + 0.004 * s[t] * sum(s[-30:]) + 1.5 * u[-30] * u[t] + 0.201
    s.append(tmps)

# plt.plot(s)
# plt.show()


class ESN:
    def __init__(self, inSize=1, resSize=100, outSize=1):
        self.inSize = inSize
        self.resSize = resSize
        self.outSize = outSize
        self.win = np.random.rand(resSize, 1 + inSize) - 0.5
        self.w = np.random.rand(resSize, resSize) - 0.5
        rhoW = max(abs(np.linalg.eigvals(self.w)))
        self.w *= 1.25 / rhoW
        self.r = np.zeros((self.resSize, 1))
        self.alpha = 0.3
        self.beta = 1e-8

    def train(self, data, initLen=100, trainLen=1000):
        X = np.zeros((1 + self.inSize + self.resSize, trainLen - initLen))
        Y = data[initLen + 1:trainLen + 1]
        Y = np.asarray(Y).reshape([1, -1])

        for i in range(trainLen):
            u = data[i]
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(
                np.dot(self.w, self.r) + np.dot(self.win, np.vstack((1, u))))
            if i >= initLen:
                X[:, i - initLen] = np.squeeze(np.vstack((1, u, self.r)))

        X_T = X.T
        self.wout = np.dot(np.dot(Y, X_T),
                           np.linalg.inv(np.dot(X, X_T) + self.beta * np.eye(1 + self.inSize + self.resSize)))

#build models with sliding window size 500
weights = []
y = []
for i in range(500):
    data = s[i:i+501] #sliding window size = 500
    esn = ESN()
    esn.train(data,initLen=50,trainLen=500)
    weights.append(esn.wout)
    y.append(0)
    print(i)
for j in range(500):
    data = s[j+2000:j + 2501]  # sliding window size = 500
    esn = ESN()
    esn.train(data, initLen=50, trainLen=500)
    weights.append(esn.wout)
    y.append(1)
    print(j)

#multi-dimensional scaling,pre-compute dissimilarity matrix
dissimilarity_matrix = np.zeros((len(weights),len(weights)))
for i,w1 in enumerate(weights):
    for j,w2 in enumerate(weights):
        if j>i:
            continue
        dissimilarity_matrix[i,j] = np.sqrt(np.sum(np.square(w1-w2)))
        dissimilarity_matrix[j,i] = dissimilarity_matrix[i,j]
# dmax = np.max(dissimilarity_matrix)
# dissimilarity_matrix = dissimilarity_matrix / dmax
print(dissimilarity_matrix)
from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
mds = MDS(n_components=3,dissimilarity='precomputed')
result = mds.fit_transform(dissimilarity_matrix) #shape = [n_samples,n_components]
fig = plt.figure()
# set up the axes for the first plot
ax = fig.add_subplot(projection='3d')

ax.scatter3D(result[:, 0], result[:, 1], result[:, 2], c=y, cmap=plt.cm.get_cmap('jet', 10))
plt.show()
