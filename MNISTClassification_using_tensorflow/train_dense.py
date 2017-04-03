import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model
from keras import optimizers



def train(trainX, trainY):
    '''
    Complete this function.

    '''
    trainX = trainX.reshape(-1,784)
    trainY = trainY.reshape(60000,1)


    tr_Y = np.zeros((60000, 10))
    for i in range(tr_Y.shape[0]):
        tr_Y[i,trainY[i]] = 1

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=784))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    Adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam,
                  metrics=['accuracy'])

    model.fit(trainX, tr_Y,
              epochs=100,
              batch_size=100)
    model.save('weights/dnn.h5')
    

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
    testX = testX.reshape(-1,784)
    model = load_model('weights/dnn.h5')

    return np.argmax(model.predict(testX),1)
