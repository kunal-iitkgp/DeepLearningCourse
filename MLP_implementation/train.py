'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:kunal singh
Roll No.: 14bt30010

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np

def softmax(x):
    x_max = np.max(x, axis=0)
    r = np.exp(x - x_max)
    return r / np.sum(r, axis=0)

class Network(object):

    def __init__(self, il, nl, sl):
        self.il = il
        self.nl = nl
        self.sl = sl
        self.b1 = np.zeros((nl, 1))
        self.b2 = np.zeros((sl, 1))
        self.w1 = np.random.uniform(-1/il**0.5, 1/il**0.5, (nl, il))
        self.w2 = np.random.uniform(-1/nl**0.5, 1/nl**0.5, (sl, nl))

    def fprop(self,x):
        z1 = np.dot(self.w1,x) + self.b1
        a1 = (z1>0)
        z2 = np.dot(self.w2,a1) + self.b2
        y_cap = softmax(z2)
        return (z1, a1, z2, y_cap)

    def bprop(self, x, y, z1, a1, z2, y_cap):
        grad_z2 = y_cap.copy()
        grad_z2[y] -= 1
        grad_b2 = grad_z2
        grad_w2 = np.dot(grad_z2, a1.T)
        grad_a1 = np.dot(self.w2.T,grad_z2)
        grad_z1 = (a1>0)*grad_a1
        grad_b1 = grad_z1
        grad_w1 = np.dot(grad_z1,x.T)

        return (grad_b1, grad_w1, grad_w2, grad_b2)

    def gradient_batch(self, x, y):
        n = x.shape[0]
        #print "n {0}".format(n)
        sum_w1 = np.zeros(self.w1.shape)
        sum_w2 = np.zeros(self.w2.shape)
        sum_b1 = np.zeros(self.b1.shape)
        sum_b2 = np.zeros(self.b2.shape)

        for i in range(n):
            #x[i] = x[i].flatten()'
            
            #print x[i].shape
            (z1, a1, z2, y_cap) = self.fprop(x[i].T.reshape(-1,1))
            ( grad_b1, grad_w1, grad_w2, grad_b2) = self.bprop(x[i].T.reshape(-1,1), y[i], z1, a1, z2, y_cap)
            sum_w1 += grad_w1
            sum_w2 += grad_w2
            sum_b1 += grad_b1
            sum_b2 += grad_b2

        return (sum_b1/n, sum_w1/n, sum_w2/n, sum_b2/n)



    def train(self, x, y, n_it = 28, batch_size = 100, eta = 0.008):
        for c_it in range(n_it):
            for i in range(int(x.shape[0] / batch_size)):
                n = min(batch_size, x.shape[0]-i*batch_size)
                (grad_b1, grad_w1, grad_w2, grad_b2) = self.gradient_batch(x[i*batch_size:i*batch_size+n], y[i*batch_size:i*batch_size+n])
                self.w1 -= eta * grad_w1 
                self.w2 -= eta * grad_w2 
                self.b1 -= eta * grad_b1 
                self.b2 -= eta * grad_b2 
            print c_it


def train(trainX, trainY):
    '''
    Complete this function.
    '''
    train_X = trainX.reshape((60000, 784)).astype(np.float)
    print train_X.shape 
    MLP = Network(784,125,10)
    MLP.train(train_X,trainY)
    np.save('w1.npy',MLP.w1)
    np.save('w2.npy',MLP.w2)
    np.save('b1.npy',MLP.b1)
    np.save('b2.npy',MLP.b2)

def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    test_x = testX.reshape((10000,784)).astype(np.float)

    MLP = Network(784,125,10)
    MLP.w1 = np.load('w1.npy')
    MLP.w2 = np.load('w2.npy')
    MLP.b1 = np.load('b1.npy')
    MLP.b2 = np.load('b2.npy')

    labels = np.zeros(test_x.shape[0])
    for i in range(test_x.shape[0]):
        (z1, a1, z2, y_cap) = MLP.fprop(test_x[i].T.reshape((-1,1)))
        labels[i] = np.argmax(y_cap.T)

    return labels
