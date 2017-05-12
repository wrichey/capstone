# code for training and testing data samples
import glob
import os
import numpy as np
import cv2
import ntpath
import tensorflow.python.platform
import tensorflow as tf
import requests
import csv
from PIL import Image
from sklearn import svm
import random

def get_accuracy(Training_data, Training_labels, negative_test_featues, positive_test_features):
        clf.fit(Training_data, Training_labels)

        true_negatives=clf.predict(negative_test_features)
        
        #there should be no nonzero elements in negative accuracy (all should be predicted as 0);   ideal: sum == 0
        #number of samples incorrectly calssified
        true_negatives=np.count_nonzero(np.array(true_negatives))
        
        #number of samples classified correctly
        tn = len(negative_test_features)-true_negatives

        #there should be no zero elements in positive accuracy (all should be predicted as 1);      ideal: len == sum
        true_positives=clf.predict(positive_test_features)

        #this is the number of samples calssified correctly
        tp=np.count_nonzero(np.array(true_positives)) 
        
        # TN + FP = all real negatives
        fp = float(len(negative_test_features)) - float(tn)
        specificity= str(float(tn)/float(len(negative_test_features))*100) #(tn)/(all real negatives)

        # TP + FN = all real positives
        fn = str(float(len(positive_test_features)) - float(tp)) # all positives - true positives
        sensitivity = str(float(tp)/float(len(positive_test_features))*100) # TP/(all real positives

        #accuracy = (number correctly classified negatives + number of correctly classified positives)/ total samples
        accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))
        return tp, tn, fp, fn, sensitivity, specificity, accuracy

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
        
def label_divide(features,names):
        pos_features = []
        pos_names = []
        neg_features = []
        neg_names = []
        for x in range(len(features)):
                feature = features[x]
                name = names[x]
                if "pos" in name:
                        pos_features.append(feature)
                        pos_names.append(name)
                else: #negative
                        neg_features.append(feature)
                        neg_names.append(name)
        return pos_features, neg_features

feature_set = "medium"
total_transfers = 24

features =np.load(feature_set + '_cell_featurevectors_DL.npy')
names = np.load(feature_set + '_cell_filenames.npy')

#features = np.squeeze(features)


for hold_x_out in [1,3,5]:#the number of images to hold out for each test (defines the training set)
        newfile = feature_set + '_linear_SVM_cell_resultsL'+ str(hold_x_out) + 'O.csv' #where results will be printed
        f8 = open(newfile, 'wb')
        fw8 = csv.writer(f8)
        
        #write file header; column headings
        test_percent= float(hold_x_out)*100/total_transfers
        fw8.writerow(['Percentage held for testing: ', test_percent])
        fw8.writerow(['Test Set starts at:','Kernel', 'deg', "True Positive", 'True Negative', 'False Positive', 'False Negative', 'Sensitivity-PosAcc', 'Specificity-NegAcc', 'Total Accuracy', 'whole image accuracy'])

        #define test features by whole image
        #       --> ex: test set = all images derived from image one
        #       each whole image produces
        for x in range(1,25): #testing for each image
                train_features, train_names, test_features, test_names, whole_features, whole_names = test_divide(features,names, x,hold_x_out)
                        
                positive_features, negative_features = label_divide(train_features,train_names)
                positive_test_features, negative_test_features = label_divide(test_features,test_names)
                whole_test_positives, whole_test_negative = label_divide(whole_features, whole_names)

                negative_labels=np.ones(len(np.array(negative_features)))*(0)
                positive_labels=np.ones(len(np.array(positive_features)))*(1)
                Training_data=[]
                Training_data=np.concatenate((positive_features,negative_features), axis=0)
                Training_labels=np.concatenate((positive_labels, negative_labels), axis=0)
        
                #kernels = ["rbf","linear", "poly","poly","poly","poly","poly"]
                kernels = ['linear']
                deg = 2
                for SVM_kern in kernels:
                        if SVM_kern == "poly":
                                clf = svm.SVC(kernel = SVM_kern, degree = deg)
                                tp, tn, fp, fn, sensitivity, specificity, accuracy= get_accuracy(Training_data, Training_labels, negative_test_features, positive_test_features)
                                fw8.writerow(['transfer_' + str(x), SVM_kern, deg, tp, tn, fp, fn, sensitivity, specificity, accuracy])
                                tp, tn, fp, fn, sensitivity, specificity, whole_img_accuracy = get_accuracy(Training_data, Training_labels, whole_test_negative, whole_test_positive)
                                fw8.writerow(["Whole image Metrics for transfer: " + str(x), SVM_kern, deg, tp, tn, fp, fn, sensitivity, specificity, accuracy])
                                
                                deg +=1
                        else:
                                clf = svm.SVC(kernel = SVM_kern)                                tp, tn, fp, fn, sensitivity, specificity, accuracy = get_accuracy(Training_data, Training_labels, negative_test_features, positive_test_features)
                                fw8.writerow(['transfer_' + str(x), SVM_kern, 'n/a' , tp, tn, fp, fn, sensitivity, specificity, accuracy])
                                
                                tp, tn, fp, fn, sensitivity, specificity, whole_img_accuracy = get_accuracy(Training_data, Training_labels, whole_test_negative, whole_test_positive)
                                fw8.writerow(["Whole image Metrics for transfer: " + str(x), SVM_kern, deg, tp, tn, fp, fn, sensitivity, specificity, accuracy])

