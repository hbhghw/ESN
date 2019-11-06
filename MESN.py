import numpy as np
from sklearn.linear_model import Ridge

class MESN:
    '''
    multiscale input ESN
    '''

    def __init__(self, inSize=1, resSize=100, outSize=1, spectral_radius=0.99, connection=0.25):
        self.inSize = inSize
        self.resSize = resSize
        self.outSize = outSize # usually outSize equals inSize
        self.win = np.random.rand(1 + inSize, resSize) - 0.5
        self.w = np.random.rand(resSize, resSize) - 0.5
        rhoW = max(abs(np.linalg.eigvals(self.w)))
        self.w *= spectral_radius / rhoW

        mask = np.random.rand(*self.w.shape)
        mask = np.where(mask < connection, 1, 0)
        self.w = self.w * mask

        self.alpha = 0.3
        self.beta  = 1e-8
        self.ridge = Ridge(alpha=1e-8,fit_intercept=True)

    def fit(self, data, trainLen=None, initLen=0, norm=None):
        '''
        transform signals data to models
        :param data: [N,T,D], N samples with time step T ,input dimension is D
        :param initLen: washout length
        :param trainLen: total train length,default len(data) - 1
        :param norm:
        :return:model weights
        '''
        assert hasattr(data, 'shape')

        if len(data.shape) < 2:
            raise ValueError("data shape should not be less than 2")
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=-1)

        N, T, D = data.shape
        assert D == self.inSize

        if trainLen is None or trainLen>=T:
            trainLen = T - 1

        self.r = np.zeros((N, self.resSize))  # reservoir state
        X = np.zeros((N, trainLen - initLen, 1 + self.inSize + self.resSize))
        Y = data[:, initLen + 1:trainLen + 1, :]  # predict for the next input

        for i in range(trainLen):
            u = data[:, i, :]  # [N,D]
            u = np.concatenate([u, np.ones((N, 1))], axis=-1)  # [N,D+1] , add bias
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(
                np.dot(self.r, self.w) + np.dot(u, self.win))
            if i >= initLen:
                X[:, i - initLen,:] = np.concatenate((u,self.r),axis=-1)

        self.model_weights = [] # representation of model's weights, transform signal space to model space
        for i in range(N):
            x = X[i]
            y = Y[i]
            wout_i = np.dot(np.linalg.inv(np.dot(x.T,x)+self.beta*np.eye(1+self.inSize+self.resSize)),np.dot(x.T,y))
            # ### Also wout_i can be computed as follow:
            # >>> self.ridge.fit(x, y)
            # >>> coef = self.ridge.coef_.ravel()
            # >>> intercept = self.ridge.intercept_.ravel()
            # >>> wout_i = np.concatenate((coef,intercept),axis=-1)
            if norm == '1':  # min-max
                maxi = np.max(wout_i)
                mini = np.min(wout_i)
                wout_i = (wout_i- mini) / (maxi - mini)
            elif norm == '2':  # Gauss
                mean = np.mean(wout_i)
                var = np.var(wout_i)
                wout_i = (wout_i - mean) / var
            self.model_weights.append(wout_i) #wout_i shape:[(1+inSize+resSize,outSize)]

        return self.model_weights



class CRJ1(ESN):
    inSize = 1
    outSize = 1
    pi = '3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067' \
         '98214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196'

    def __init__(self, ri, rc, rj, jump_len, ri_sign=None, resSize=100, alpha=0.3, beta=1e-8):
        super(CRJ1, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.resSize = resSize
        self.ri_sign = ri_sign if ri_sign else CRJ1.pi

        # init w
        self.w = np.zeros((resSize, resSize))
        for i in range(resSize - 1):
            self.w[i + 1, i] = rc
        self.w[0, resSize - 1] = rc
        for i in range(0, resSize - jump_len + 1, jump_len):
            self.w[i, (i + jump_len) % resSize] = rj
            self.w[(i + jump_len) % resSize, i] = rj
        # rhoW = max(abs(np.linalg.eigvals(self.w)))
        # self.w *= 1.25 / rhoW

        # init w_in
        self.win = np.zeros((resSize, 1 + self.inSize))
        for i in range(resSize):
            if self.ri_sign[i + 2] < '5':
                self.win[i, 0] = -ri
            else:
                self.win[i, 0] = ri




if __name__ == '__main__':
    from scipy.io import loadmat
    data = loadmat('JpVow.mat')
    X = data['X']  # shape  [N,T,D]
    Y = data['Y']  # shape  [N,1]
    Y = Y.reshape((-1,))
    Xte = data['Xte']
    Yte = data['Yte']
    Yte = Yte.reshape((-1,))

    esn = ESN(inSize=X.shape[-1])

    train_weights = esn.fit(X,norm='2')
    train_weights = [wout.reshape((-1,)) for wout in train_weights]

    test_weights = esn.fit(Xte,norm='2')
    test_weights = [wout.reshape((-1,)) for wout in test_weights]

    #from tensorflow import keras
    #model = build_model()
    #model.train(train_weights,Y)
    #pred = model.predict(test_weights,Yte)
    #print(np.mean(pred==Yte))




