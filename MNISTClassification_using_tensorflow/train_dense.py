'''
Deep Learning Programming Assignment 2
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf
import pickle

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def train(trainX, trainY):
    trainX = trainX.reshape(-1,784)
    trainY = trainY.reshape(60000,1)


    tr_Y = np.zeros((60000, 10))
    for i in range(tr_Y.shape[0]):
        tr_Y[i,trainY[i]] = 1

    '''te_Y = np.zeros((10000, 10))
    for i in range(te_Y.shape[0]):
        te_Y[i,testY[i]] = 1'''


    learning_rate = 0.0005
    training_epochs = 150
    batch_size = 100
    display_step = 1

    n_hidden_1 = 246 # 1st layer number of features
    n_hidden_2 = 246 # 2nd layer number of features
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])


    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }    

    pred = multilayer_perceptron(x, weights, biases)
    print pred.shape
    #saver=tf.train.Saver()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(trainX.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = trainX[i*batch_size:(i+1)*batch_size,:], tr_Y[i*batch_size:(i+1)*batch_size,:]    
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: testX, y: te_Y}))

        savepath = saver.save(sess,'model_dnn.ckpt')
        sess.close()
    


    '''
    Complete this function.
    '''


'''def test(testX):
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

    n_hidden_1 = 246 # 1st layer number of features
    n_hidden_2 = 246 # 2nd layer number of features
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }    

    

    pred = multilayer_perceptron(x, weights, biases)

    with tf.Session() as sess1:
    # Restore variables from disk.
        sess1.run(tf.global_variables_initializer())
        with open('weights_dnn.pkl','rb') as f:
            weights['h1'] = pickle.load(f)
            weights['h2'] = pickle.load(f)
            weights['out'] = pickle.load(f)
        with open('biases_dnn.pkl','rb') as f:
            biases['h1'] = pickle.load(f)
            biases['h2'] = pickle.load(f)
            biases['out'] = pickle.load(f) 
        label = pred.eval(feed_dict = {x: testX})
        print label.shape

        label = np.argmax(label,1)

        print("Model restored.")

    sess1.close()
    return label'''
