import numpy as np

x_data = np.zeros((380,137,569))
y_data = np.zeros((380))

x_data = np.load("../processed_voxels.npy")
y_data = np.load("../processed_labels.npy")

# print x_data.shape,y_data
def data_format():
	return x_train,y_train,x_test,y_test

def main():
	trainX, trainY, testX, testY = data_format()
	print "Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape

	train_lstm.train(trainX,trainY)
	labels = train_lstm.test(testX)
    accuracy = np.mean((labels == testY)) * 100.0
    print "\nTest accuracy: %lf%%" % accuracy

