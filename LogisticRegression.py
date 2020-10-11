class LogisticRegression:
    def __init__(self):
        self.w = np.array([])
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.err_list = []
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # lr: gradient descent step
    # max_iter: the max iterations to run
    def fit(self, X, y, lr=0.001, max_iter=500):
        # process the data
        X_mat = np.mat(X)
        y_mat = np.mat(y).transpose()
        m, n = X_mat.shape
        self.w = np.ones((n, 1))
        self.err_list = []
        # do gradient descent
        for i in range(max_iter):
            sig = self.sigmoid(X_mat * self.w)
            err = y_mat - sig
            self.err_list.append(np.linalg.norm(err))
            grad = X_mat.transpose() * err
            if np.linalg.norm(grad) < 3:
                break
            self.w += lr * grad
        
        
    # threshold: the boundary for probability to classify the data
    def predict(self, X, y, threshold=0.5):
        # calculate the probability
        prob = self.sigmoid(X.dot(self.w))
        # predict
        y_pred = (prob > threshold).reshape(1, -1).astype(int)
        # calculate availability
        self.accuracy = (y_pred == y).sum() / len(y)
        TP = ((y_pred == 1).astype(int) + (y == 1).astype(int) == 2).sum()
        TN = ((y_pred == 0).astype(int) + (y == 0).astype(int) == 2).sum()
        FP = ((y_pred == 1).astype(int) + (y == 0).astype(int) == 2).sum()
        FN = ((y_pred == 0).astype(int) + (y == 1).astype(int) == 2).sum()
        self.precision = TP / (TP + FP)
        self.recall = TP / (TP + FN)
        
        return y_pred
        
