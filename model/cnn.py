import numpy as np

class MultiClassClassification:

    def __init__(self, lr=0.01):

        self.lr = lr

        fan_in_conv = 3 * 3
        std_conv = np.sqrt(2.0 / fan_in_conv)

        self.K = np.random.randn(3, 3) * std_conv
        self.b_conv = 0.0

        fan_in_fc = 13 * 13
        std_fc = np.sqrt(2.0 / fan_in_fc)

        self.W_fc = np.random.randn(10, 169) * std_fc
        self.b_fc = np.zeros((10, 1))

    def conv_forward(self, X):
        H, W = X.shape
        kH, kW = self.K.shape

        out_H = H - kH + 1
        out_W = W - kW + 1

        Z = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                patch = X[i:i+kH, j:j+kW]
                Z[i, j] = np.sum(patch * self.K) + self.b_conv

        return Z

    def relu_forward(self, Z):
        return np.maximum(0, Z)

    def maxpool_forward(self, A):
        H, W = A.shape
        pool_size = 2
        stride = 2

        out_H = H // 2
        out_W = W // 2

        P = np.zeros((out_H, out_W))
        mask = np.zeros_like(A)

        for i in range(out_H):
            for j in range(out_W):

                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size

                patch = A[h_start:h_end, w_start:w_end]
                max_val = np.max(patch)
                P[i, j] = max_val

                max_index = np.argmax(patch)
                r, c = divmod(max_index, pool_size)
                mask[h_start + r, w_start + c] = 1

        return P, mask

    def flatten_forward(self, P):
        return P.reshape(-1, 1)

    def linear_forward(self, x):
        return np.dot(self.W_fc, x) + self.b_fc

    def softmax(self, z):
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)

    def forward(self, X):

        self.X = X

        self.Z_conv = self.conv_forward(X)
        self.A = self.relu_forward(self.Z_conv)
        self.P, self.pool_mask = self.maxpool_forward(self.A)
        self.x = self.flatten_forward(self.P)
        self.z = self.linear_forward(self.x)
        self.y_pred = self.softmax(self.z)

        return self.y_pred

    def backward(self, y_true):

        dz = self.y_pred - y_true

        # Linear Backward
        dW_fc = np.dot(dz, self.x.T)
        db_fc = dz
        dx = np.dot(self.W_fc.T, dz)

        # Reshape
        dP = dx.reshape(13, 13)

        # MaxPool Backward
        dA = np.zeros_like(self.A)

        for i in range(13):
            for j in range(13):

                h_start = i * 2
                h_end = h_start + 2
                w_start = j * 2
                w_end = w_start + 2

                dA[h_start:h_end, w_start:w_end] += (
                    self.pool_mask[h_start:h_end, w_start:w_end] * dP[i, j]
                )

        # ReLU Backward
        dZ = dA.copy()
        dZ[self.Z_conv <= 0] = 0

        # Conv Backward
        dK = np.zeros_like(self.K)
        dX = np.zeros_like(self.X)
        db_conv = 0.0

        for i in range(26):
            for j in range(26):

                patch = self.X[i:i+3, j:j+3]
                grad = dZ[i, j]

                dK += patch * grad
                dX[i:i+3, j:j+3] += self.K * grad
                db_conv += grad

        # Store gradients
        self.dW_fc = dW_fc
        self.db_fc = db_fc
        self.dK = dK
        self.db_conv = db_conv


    # Update
    def update(self):

        self.W_fc -= self.lr * self.dW_fc
        self.b_fc -= self.lr * self.db_fc

        self.K -= self.lr * self.dK
        self.b_conv -= self.lr * self.db_conv
