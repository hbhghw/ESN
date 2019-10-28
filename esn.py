# Echo state network

import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0, 50, 6000)
b = np.sin(2 * np.pi * a)

initLen = 100
trainLen = 3000
testLen = 2000
# plt.plot(a,b)
plt.plot(a[:trainLen], b[:trainLen], 'b')  # end at point (trainLen-1)

inSize = 1
resSize = 100
alpha = 0.3
reg = 1e-8

#w and w_in are random initialized
win = np.random.rand(resSize, 1 + inSize) - 0.5

w = np.random.rand(resSize, resSize) - 0.5
rhoW = max(abs(np.linalg.eigvals(w)))
w *= 1.25 / rhoW

X = np.zeros((1 + inSize + resSize, trainLen - initLen))
Y = b[initLen + 1:trainLen + 1]
r = np.zeros((resSize, 1))
for i in range(trainLen):
    u = b[i]  # input
    r = (1 - alpha) * r + alpha * np.tanh(np.dot(win, np.vstack((1, u))) + np.dot(w, r))
    if i >= initLen:
        X[:, i - initLen] = np.vstack((1, u, r))[:, 0]

X_T = X.T
wout = np.dot(np.dot(Y, X_T), np.linalg.inv(np.dot(X, X_T) + reg * np.eye(1 + inSize + resSize)))

predict = [b[trainLen - 1], b[trainLen]]  # start from point (trainLen - 1)
pred_x = [a[trainLen - 1], a[trainLen]]
u = b[trainLen]  # predict from point (trainLen+1)
for i in range(testLen):
    r = (1 - alpha) * r + alpha * np.tanh(np.dot(win, np.vstack((1, u))) + np.dot(w, r))
    out = np.dot(wout, np.vstack((1, u, r)))
    out = np.squeeze(out)
    u = out  # next state's input
    predict.append(out)
    pred_x.append(a[trainLen + i + 1])

plt.plot(pred_x, predict, 'r')
plt.show()
