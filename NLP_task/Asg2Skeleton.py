
# coding: utf-8

# Deep Learning Programming Assignment 2
# --------------------------------------
# Name:kunal singh
# Roll No.:14bt30010
# 
# Submission Instructions:
# 1. Fill your name and roll no in the space provided above.
# 2. Name your folder in format <Roll No>_<First Name>.
#     For example 12CS10001_Rohan
# 3. Submit a zipped format of the file (.zip only).
# 4. Submit all your codes. But do not submit any of your datafiles
# 5. From output files submit only the following 3 files. simOutput.csv, simSummary.csv, analogySolution.csv
# 6. Place the three files in a folder "output", inside the zip.

# In[59]:
import random
import gzip
import os
import csv
import numpy as np
from scipy import spatial
import tensorflow as tf 
from sklearn.model_selection import KFold
import random
import sys





## paths to files. Do not change this
simInputFile = "Q1/word-similarity-dataset"
analogyInputFile = "Q1/word-analogy-dataset"
vectorgzipFile = "Q1/glove.6B.300d.txt.gz"
vectorTxtFile = "Q1/glove.6B.300d.txt"   # If you extract and use the gz file, use this.
analogyTrainPath = "Q1/wordRep/"
simOutputFile = "Q1/simOutput.csv"
simSummaryFile = "Q1/simSummary.csv"
anaSOln = "Q1/analogySolution.csv"
Q4List = "Q4/wordList.csv"




# In[ ]:

# Similarity Dataset
simDataset = [item.split(" | ") for item in open(simInputFile).read().splitlines()]
#print simDataset
# Analogy dataset
analogyDataset = [[stuff.strip() for stuff in item.strip('\n').split('\n')] for item in open(analogyInputFile).read().split('\n\n')]

def vectorExtract(simD = simDataset, anaD = analogyDataset, vect = vectorgzipFile):
    simList = [stuff for item in simD for stuff in item]
    analogyList = [thing for item in anaD for stuff in item[0:4] for thing in stuff.split()]
    simList.extend(analogyList)
    #print simList
    wordList = set(simList)
        #print("wordlist")
    #print len(wordList)
    wordDict = dict()
    
    vectorFile = gzip.open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = line.split()[1:]
    
    
    vectorFile.close()
    print 'retrieved', len(wordDict.keys())
    return wordDict

# Extracting Vectors from Analogy and Similarity Dataset
validateVectors = vectorExtract()
#print validateVectors


# In[ ]:

# Dictionary of training pairs for the analogy task
trainDict = dict()
for subDirs in os.listdir(analogyTrainPath):
    for files in os.listdir(analogyTrainPath+subDirs+'/'):
        f = open(analogyTrainPath+subDirs+'/'+files).read().splitlines()
        trainDict[files] = f
print len(trainDict.keys())
print ("train")


def similarityTask(inputDS = simDataset, outputFile = simOutputFile, summaryFile=simSummaryFile, vectors=validateVectors):
    print 'hello world'
    scsv = open('Q1/simOutput.csv','w') 
    columns_s = ['file_line-number', 'query word, option word', 'distance metric', 'similarity score' ]
    writer = csv.DictWriter(scsv, fieldnames = columns_s)
    
    writer.writeheader()
    count = 0
    cc = 0
    mrr = 0
    row = []
    func = ['C','E','M']
    for sel in func:
        count = 0
        cc = 0
        mrr = 0
        for item in inputDS:
            flag = 0
            #check validity
            for x in xrange(0,5):
                if item[x] not in vectors.keys():
                    flag = 1
                    continue
            if flag == 1:
                continue
            count+=1 
            #done
            #score compu
            rank = 1.0
            query = item[0]
            List_q = vectors[query]
            a_q = np.array(List_q,dtype=float)
            List_option = vectors[item[1]]
            a_option = np.array(List_option,dtype=float)
            if sel == 'C':
                score_1 = 1 - spatial.distance.cosine(a_q, a_option)
                writer.writerow({'file_line-number':count, 'query word, option word':(query,item[1]), 'distance metric': 'C', 'similarity score':score_1})
                for i in xrange(2,5):
                    List_option = vectors[item[i]]
                    a_option = np.array(List_option,dtype=float)
                    score = 1 - spatial.distance.cosine(a_q, a_option)
                    #print score
                    writer.writerow({'file_line-number':count, 'query word, option word':(query,item[i]), 'distance metric': 'C', 'similarity score':score})
                    if score>score_1:
                        #print maxs
                        rank+=1
                        #print rank
                if rank == 1:
                    cc+=1
                mrr += (1.0/rank)

                #print mrr

            elif sel == 'E':
                score_1 = spatial.distance.euclidean(a_q, a_option)
                writer.writerow({'file_line-number':count, 'query word, option word':(query,item[1]), 'distance metric': 'E', 'similarity score':score_1})
                for i in xrange(2,5):
                    List_option = vectors[item[i]]
                    a_option = np.array(List_option,dtype=float)
                    score =  spatial.distance.euclidean(a_q, a_option)
                    writer.writerow({'file_line-number':count, 'query word, option word':(query,item[i]), 'distance metric': 'E', 'similarity score':score})
                    #print score
                    if score<score_1:
                        #print maxs
                        rank+=1
                        #print rank
                if rank == 1:
                    cc+=1
                mrr += (1.0/rank)
                #print mrr

            else:
                score_1 =  spatial.distance.cityblock(a_q, a_option)
                writer.writerow({'file_line-number':count, 'query word, option word':(query,item[1]), 'distance metric': 'M', 'similarity score':score_1})
                for i in xrange(2,5):
                    List_option = vectors[item[i]]
                    a_option = np.array(List_option,dtype=float)
                    score = spatial.distance.cityblock(a_q, a_option)
                    writer.writerow({'file_line-number':count, 'query word, option word':(query,item[i]), 'distance metric': 'M', 'similarity score':score})
                    #print score
                    if score<score_1:
                        #print maxs
                        rank+=1
                        #print rank
                if rank == 1:
                    cc+=1
                mrr += (1.0/rank)
                #print mrr
        mrr = mrr/count
            #print mrr
        dic = {'Distance_Metric':sel, 'Number_of_questions_which_are_correct':cc, 'Total_questions_evalauted':count, 'MRR':mrr}
        row.append(dic)
    scsv = open('Q1/simSummary.csv','w') 
    columns_s = ['Distance_Metric', 'Number_of_questions_which_are_correct', 'Total_questions_evalauted', 'MRR']
    writer = csv.DictWriter(scsv, fieldnames = columns_s)
    writer.writeheader()
    writer.writerow(row[0])
    writer.writerow(row[1])
    writer.writerow(row[2])

    """
    Output simSummary.csv in the following format
    Distance Metric, Number of questions which are correct, Total questions evalauted, MRR
    C, 37, 40, 0.61
    """

    """
    Output a CSV file titled "simOutput.csv" with the following columns

    file_line-number, query word, option word i, distance metric(C/E/M), similarity score 

    For the line "rusty | corroded | black | dirty | painted", the outptut will be

    1,rusty,corroded,C,0.7654
    1,rusty,dirty,C,0.8764
    1,rusty,black,C,0.6543


    The order in which rows are entered does not matter and so do Row header names. Please follow the order of columns though.
    """


# In[ ]:
def train_data_generation():
    print ("generating traindata")
    def vectorExtract(path = analogyTrainPath, vect = vectorgzipFile):
        # simList = [stuff for item in simD for stuff in item]
        # analogyList = [thing for item in anaD for stuff in item[0:4] for thing in stuff.split()]
        # simList.extend(analogyList)
        simList = []
        for x in trainDict:
            for pair in trainDict[x]:
                pair_w = pair.split('\t')
                simList.append(pair_w[0])
                simList.append(pair_w[1])
            
        
        # simList = [for stuff in ]
        wordList = set(simList)
        print len(wordList)
        wordDict = dict()
        
        vectorFile = gzip.open(vect, 'r')
        for line in vectorFile:
            if line.split()[0].strip() in wordList:
                wordDict[line.split()[0].strip()] = line.split()[1:]
        
        
        vectorFile.close()
        return wordDict

    # Extracting Vectors from Analogy and Similarity Dataset
    vectors = vectorExtract()


    analogyTrainInput = []
    analogyTrainOutput = []
    count = 0
    while (count < 60000):
        x = random.choice(trainDict.keys())
        pair = random.choice(trainDict[x])
        pair_w = pair.split('\t')
        simip = []
        try:
            simip.extend(vectors[pair_w[0]])
            simip.extend(vectors[pair_w[1]])
        except KeyError:
            continue
        while(True):
            simtemp = []
            try:
                similarpair = random.choice(trainDict[x])
                while (similarpair == pair):
                    similarpair = random.choice(trainDict[x])
                similarpair_w = similarpair.split('\t')
                simtemp.extend(vectors[similarpair_w[0]])
                simtemp.extend(vectors[similarpair_w[1]])
                break
            except KeyError:
                pass
        simip.extend(simtemp)
        templist = [simip]
        for i in range(4):
            temp = []
            while(True):
                try:
                    temp2 = []
                    randfile = random.choice(trainDict.keys())
                    while (randfile == x):
                        randfile = random.choice(trainDict.keys())
                    randpair = random.choice(trainDict[randfile])
                    randpair_w = randpair.split('\t')

                    temp2.extend(vectors[pair_w[0]])
                    temp2.extend(vectors[pair_w[1]])
                    temp2.extend(vectors[randpair_w[0]])
                    temp2.extend(vectors[randpair_w[1]])
                    break
                except KeyError:
                    pass
            temp = temp2

            templist.append(temp)

            

        random.shuffle(templist)
        singleinput = [item for sublist in templist for item in sublist]

        singleip_np = np.array(singleinput,dtype=np.float64)
        analogyTrainInput.append(singleip_np)
        tout = [0,0,0,0,0]
        for i in range(5):
            if templist[i] == simip:
                tout[i] = 1
                tout_np = np.array(tout)
                analogyTrainOutput.append(tout_np)
                break
        count = count + 1


    analogyTrainInput_np = np.array(analogyTrainInput)

    analogyTrainOutput_np = np.array(analogyTrainOutput)
    print analogyTrainInput_np.shape
    print analogyTrainOutput_np.shape


    np.save('train_X.npy',analogyTrainInput_np)
    np.save('train_Y.npy',analogyTrainOutput_np)

def test_data_generation():
    print ("generating test data")
    testdata = []
    testdataop = []
    for item in analogyDataset:
        try:
            temp = []
            for i in range(5):
                temp1 = item[0].split(' ')
                temp2 = item[i+1].split(' ')
                # temp.extend([temp1[0]])
                # temp.extend([temp1[1]])
                # temp.extend([temp2[0]])
                # temp.extend([temp2[1]])
                temp.extend(validateVectors[temp1[0]])
                temp.extend(validateVectors[temp1[1]])
                temp.extend(validateVectors[temp2[0]])
                temp.extend(validateVectors[temp2[1]])
            temp_np = np.array(temp,dtype=np.float64)
            testdata.append(temp_np)
            testop = [0,0,0,0,0]
            testop[ord(item[6]) - ord('a')] = 1
            testop_np = np.array(testop)
            testdataop.append(testop_np)
        except KeyError:
            continue
    testdata_np = np.array(testdata)
    testdataop_np = np.array(testdataop)


    np.save('test_X.npy',testdata_np)
    np.save('test_Y.npy',testdataop_np)


def analogyTask(inputDS=analogyDataset ):
    train_data_generation()
    test_data_generation()
     # add more arguments if required
    X_train=np.load("train_X.npy")
    y_train=np.load("train_Y.npy")
    testtX=np.load("test_X.npy")
    testtY=np.load("test_Y.npy")
    print X_train.shape


    learning_rate = 0.001
    training_epochs = 30
    batch_size = 100
    display_step = 1

    n_hidden = 600 # 1st layer number of features
    # n_hidden_2 = 256 # 2nd layer number of features
    n_input = 1200 # MNIST data input (img shape: 28*28)
    n_classes = 5 # MNIST total classes (0-9 digits)

    x = tf.placeholder("float", [None, 5*n_input])
    y = tf.placeholder("float", [None, n_classes])


    def multilayer_perceptron(x, weights, b1aiases):
        # Hidden layer with RELU activation
        xa , xb, xc, xd, xe = tf.split(x,[1200,1200,1200,1200,1200],1)

        layer_1a = tf.add(tf.matmul(xa, weights['h1a']), biases['b1a'])
        layer_1a = tf.nn.relu(layer_1a)
        layer_1b = tf.add(tf.matmul(xb, weights['h1b']), biases['b1b'])
        layer_1b = tf.nn.relu(layer_1b)
        layer_1c = tf.add(tf.matmul(xc, weights['h1c']), biases['b1c'])
        layer_1c = tf.nn.relu(layer_1c)
        layer_1d = tf.add(tf.matmul(xd, weights['h1d']), biases['b1d'])
        layer_1d = tf.nn.relu(layer_1d)
        layer_1e = tf.add(tf.matmul(xe, weights['h1e']), biases['b1e'])
        layer_1e = tf.nn.relu(layer_1e)

        layer_h=tf.concat([layer_1a,layer_1b,layer_1c,layer_1d,layer_1e],1)
        # print layer_h.get_shape()

        # Hidden layer with RELU activation
        # layer_2 = tf.add(tf.matmul(layer_h, weights['h2']), biases['b2'])
        # layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_h, weights['out']) + biases['out']
        return out_layer

    weights = {
        'h1a': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'h1b': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'h1c': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'h1d': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'h1e': tf.Variable(tf.random_normal([n_input, n_hidden])),
        # 'h2': tf.Variable(tf.random_normal([5*n_hidden, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([5*n_hidden, n_classes]))
    }
    biases = {
        'b1a': tf.Variable(tf.random_normal([n_hidden])),
        'b1b': tf.Variable(tf.random_normal([n_hidden])),
        'b1c': tf.Variable(tf.random_normal([n_hidden])),
        'b1d': tf.Variable(tf.random_normal([n_hidden])),
        'b1e': tf.Variable(tf.random_normal([n_hidden])),
        # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }    


    pred = multilayer_perceptron(x, weights, biases)
    #saver = tf.train.Saver()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        kf = KFold(n_splits=5,shuffle=False)
        kf.get_n_splits(X_train)
        for train_index, test_index in kf.split(X_train):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train_1, X_test_1 = X_train[train_index], X_train[test_index]
            y_train_1, y_test_1 = y_train[train_index], y_train[test_index]

        # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.0
                
                total_batch=X_train_1.shape[0]/batch_size
                # Loop over all batches
                for i in range(total_batch):

                    batch_x, batch_y = X_train_1[i*batch_size:(i+1)*batch_size],y_train_1[i*batch_size:(i+1)*batch_size]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                # Calculate accuracykm;     
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("CV-Accuracy:", accuracy.eval({x: X_test_1, y: y_test_1}))

    #save_path = saver.save(sess, "Analogy_model.ckpt")
    #print("Saving File at: %s" % save_path)

    #saver.restore(sess,"Analogy_model.ckpt") 
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Test Accuracy:", accuracy.eval(session=sess,feed_dict={x: testtX, y: testtY}))  
            
        sess.close()


def derivedWOrdTask(inputFile = Q4List):
    return
    
    """
    Output vectors of 3 files:
    1)AnsFastText.txt - fastText vectors of derived words in wordList.csv
    2)AnsLzaridou.txt - Lazaridou vectors of the derived words in wordList.csv
    3)AnsModel.txt - Vectors for derived words as provided by the model
    
    For all the three files, each line should contain a derived word and its vector, exactly like 
    the format followed in "glove.6B.300d.txt"
    
    word<space>dim1<space>dim2........<space>dimN
    charitably 256.238 0.875 ...... 1.234
    
    """
    
    """
    The function should return 2 values
    1) Averaged cosine similarity between the corresponding words from output files 1 and 3, as well as 2 and 3.
    
        - if there are 3 derived words in wordList.csv, say word1, word2, word3
        then find the cosine similiryt between word1 in AnsFastText.txt and word1 in AnsModel.txt.
        - Repeat the same for word2 and word3.
        - Average the 3 cosine similarity values
        - DO the same for word1 to word3 between the files AnsLzaridou.txt and AnsModel.txt 
        and average the cosine simialities for valuse so obtained
        
    """
    #return cosVal1,cosVal2
    


# In[ ]:

def main():
    similarityTask()
    analogyTask()
    derivedWOrdTask()

if __name__ == '__main__':
    main()
