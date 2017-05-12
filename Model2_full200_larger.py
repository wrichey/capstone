# code for training and testing data samples
#uses multilayer perceptron -- INCOMPLETE 
import glob
import os
import numpy as np
import cv2
import ntpath
import tensorflow.python.platform
import tensorflow as tf
from scipy import sparse
import requests
import csv
from PIL import Image
from sklearn import svm
import random


def label_split(data, label_index):
    '''input a matrix, data with each row representing
       an example with a label at data[row][label_index]
       also converts labels to integers
    '''
    data_array= []
    label_array= []
    for row in data:
        t = row[label_index]
        if t == 0: #negative
            t = [0]
        else: #positive
            t = [1]
        label_array.append(t)
        del row[label_index]
        data_array.append(row)
    return data_array, label_array

def test_divide(features,names, test_image_index, hold_x_out):
        train_features = []
        train_names = []
        test_features = []
        test_names = []
        whole_features = []
        whole_names = []
        #define the names you're taking as test set
        #iterate through image indices test_image_index+[0..hold_x_out]
        image_names = []
        for x in range(hold_x_out):
                #location of filename dependent on the # of digits in the index
                image_names.append("transfer_" + str((test_image_index-1+x)%24 + 1)+ '_')

        print "Testing with hold out of whole image number: "
        print image_names

        for x in range(len(features)-1):
                feature = features[x]
                name = names[x]
                this_name = name[:(10+len(str(test_image_index)))]                                        
                if this_name in image_names:
                        test_features.append(feature)
                        test_names.append(name)
                        if 'whole' in this_name:
                                whole_features.append(feature)
                                whole_names.append(name)

                else:
                        train_features.append(feature)
                        train_names.append(name)
        return train_features, train_names, test_features, test_names, whole_features, whole_names

# from Github: Mistobaan/tensorflow_confusion_metrics.py   
# from https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data
def tf_confusion_metrics(model, actual_classes, session, feed_dict):
    predictions = tf.argmax(model, 1)
    actuals = tf.argmax(actual_classes, 1)
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
                ),
            "float"
            )
        )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
                ),
            "float"
            )
        )
    
    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
                ),
            "float"
            )
        )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
                ),
            "float"
            )
        )

    tp, tn, fp, fn = \
        session.run(
            [tp_op, tn_op, fp_op, fn_op],
            feed_dict
        )
    tpr = float(tp)/(float(tp) + float(fn))
    fpr = float(fp)/(float(tp) + float(fn))
    specificity = float(tn)/(float(tn) + float(fp))
    accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
    sensitivity = tpr
    return tp, tn, fp, fn, sensitivity, specificity, accuracy

def get_labels(features,names):
    '''
    parameters: features: feature vector np array
                names: array of image titles as a strings; names[x] = image name for features[x]
    returns: numpy array of labels, in the same order as features and string names
             where 
    
'''
    labels = []
    for x in range(len(features)):
        feature = features[x]
        name = names[x]
        if "pos" in name:
            labels.append([1,0])
        else: #negative
            labels.append([0,1])
    return np.asarray(labels)

def multilayer_perceptron(x, weights, biases):
    '''
    parameters:
    x: data, without labels, as an array
    weights: weight vectors, as an arry 
    biases: weight vector
       matmul - matrix multiplication
       add - element wise matrix addition
       --> x.w + b
'''
    # Hidden layer 
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    
    # Hidden layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    
    # Hidden layer
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.relu(layer_3)
    
    # Hidden layer 
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    
    # Output layer
    output_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return output_layer


feature_set = "larger_"
features =np.load(feature_set + 'cell_featurevectors_DL.npy')
names = np.load(feature_set + 'cell_filenames.npy')
total_transfers = 24 #the total number of images you want to train/test on (training+test set)

'''------------------------model begins-------------------'''

#parameters
training_epochs = 1000
batch_size = 10
display_step = 100

# Network Parameters
n_hidden_1 = 200 # 1st layer number of nodes
n_hidden_2 = 200 # 2nd layer number of nodes
n_hidden_3 = 200 # 2nd layer number of nodes
n_hidden_4 = 200 # 2nd layer number of nodes
n_input = len(features[0]) #  data input size (length of feature vector for each data point)
n_classes = 2 # number of labels

# tensorflow Graph input
x = tf.placeholder("float", [None, n_input]) #initializing data placeholder
y = tf.placeholder("float", [None, n_classes]) #initializind label placeholder

# Initializing layers weights & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
#need a bias for every layer
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),    
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
for hold_x_out in [5]: #the number of images to hold out for each test (defines the training set)
    newfile = feature_set + '_200_MLP_cell_results_L' + str(hold_x_out) + 'O.csv' #where results will be printed
    file = open(newfile, 'wb')
    fw8 = csv.writer(file)
    
    #write file header; parameters and column headings
    test_percent= float(hold_x_out)*100/total_transfers

    fw8.writerow(['leaning rate','max training epochs', 'batch_size', 'h1', 'h2-RELU', 'h3-RELU', 'h4', 'Percentage for Test set'])
    fw8.writerow(['cost/10', training_epochs, batch_size, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, test_percent])
    fw8.writerow(['Test Set starts at:', "True Positive", 'True Negative', 'False Positive', 'False Negative', 'Sensitivity-PosAcc', 'Specificity-NegAcc', 'Total Accuracy'])

    for transfer in range(3,4): #total_transfers+1):
        #reset learning rate
        learning_rate = float(1)
    
        #split into training and testing sets
        train_data, train_names, test_data, test_names, wholeimg_features, wholeimg_names = test_divide(features,names, transfer, hold_x_out)

        #convert names (Strings) into labels (matrices)
        train_labels = get_labels(train_data, train_names)
        test_labels = get_labels(test_data, test_names)
        wholeimg_labels = get_labels(wholeimg_features, wholeimg_names)

        # Construct model, with random weights/biases and empty x
        pred = multilayer_perceptron(x, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
    
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(train_data)/batch_size)
        
                # Loop over all batches
                for i in range(total_batch):
                    #defining where the batch is coming from
                    start = i*batch_size
                    end = i*batch_size+batch_size
                    if end > len(train_data):
                        end = len(train_data)
                        
                    batch_x = train_data[start:end]
                    batch_y = train_labels[start:end]
        
                    # Run optimization op (backpropogation) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        
                    # Compute average loss
                    avg_cost += c / total_batch
    
                    # Assign learning rate dynamically
                    learning_rate = float(avg_cost)/100
                    if learning_rate > 100:
                        learning_rate = float(100)
                    
                # Display logs per epoch step
                if (epoch + 1) % display_step == 1:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
                if avg_cost == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
                    break
        
            print("Optimization Done.")
        
            # Test model: tf_confusion_metrics(model, actual_classes, session, feed_dict):
            tp, tn, fp, fn, sensitivity, specificity, accuracy = tf_confusion_metrics(pred, y, sess, feed_dict={x: test_data, y: test_labels})            
            fw8.writerow(["transfer "+ str(transfer), tp, tn, fp, fn, sensitivity, specificity, accuracy])
            print("Accuracy: " + str(accuracy))
            
 

    file.close()


