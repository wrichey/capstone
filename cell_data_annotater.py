# code for image annotating;
# displays each image crop and lets the user write labels in terminal
# writes these labels to a csv file for each corresponding image
import csv
import glob
import os
import itk
import SimpleITK as sitk
from numpy import genfromtxt
import csv
import math
import cv2
from PIL import Image
import skimage
import skimage.io
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import string
import matplotlib.cm as cm
import ntpath
import random



# folder location of the pictures you're tryint to annotate
folder = 'pictures1'

#csv location where annotations will be saved
newfile = 'manual_crop_annotations_' + folder + '.csv'

#ending (filetype) of files you're trying to annotate
#ex: "bmp", "tiff"
imgtype = 'bmp'

f = open(newfile, 'wb')
fw = csv.writer(f)
fw.writerow(['picture','Label'])

#all_labels = []
for x in range(1,17):
    BeforeName = folder + '/' + str(x) +'A.' + imgtype
    AfterName = folder +'/' + str(x) +'B.' +imgtype
    BefIm = cv2.imread(BeforeName, 0)
    AfIm = cv2.imread(AfterName, 0)

    
    [h, w] = BefIm.shape
    #fourths
    BefIm_cut_1 = BefIm[0:int(h/2), 0:int(w/2)]
    BefIm_cut_2 = BefIm[0:int(h/2), int(w/2):w]
    BefIm_cut_3 = BefIm[int(h/2):h, 0:int(w/2)]
    BefIm_cut_4 = BefIm[int(h/2):h, int(w/2):w]
    #sixths
    BefIm_cut_5 = BefIm[0:int(h/2), 0:int(w/3)]
    BefIm_cut_6 = BefIm[0:int(h/2), int(w/3):int(2*w/3)]
    BefIm_cut_7 = BefIm[0:int(h/2), int(2*w/3):w]
    BefIm_cut_8 = BefIm[int(h/2):h, 0:int(w/3)]
    BefIm_cut_9 = BefIm[int(h/2):h, int(w/3):int(2*w/3)]
    BefIm_cut_10 = BefIm[int(h/2):h, int(2*w/3):w]
    
    before = [BefIm, BefIm_cut_1, BefIm_cut_2,BefIm_cut_3, BefIm_cut_4, BefIm_cut_5, BefIm_cut_6, BefIm_cut_7, BefIm_cut_8, BefIm_cut_9, BefIm_cut_10]
    #fourths
    AfIm_cut_1 = AfIm[0:int(h/2), 0:int(w/2)]
    AfIm_cut_2 = AfIm[0:int(h/2), int(w/2):w]
    AfIm_cut_3 = AfIm[int(h/2):h, 0:int(w/2)]
    AfIm_cut_4 = AfIm[int(h/2):h, int(w/2):w]
    #sixths
    AfIm_cut_5 = AfIm[0:int(h/2), 0:int(w/3)]
    AfIm_cut_6 = AfIm[0:int(h/2), int(w/3):int(2*w/3)]
    AfIm_cut_7 = AfIm[0:int(h/2), int(2*w/3):w]
    AfIm_cut_8 = AfIm[int(h/2):h, 0:int(w/3)]
    AfIm_cut_9 = AfIm[int(h/2):h, int(w/3):int(2*w/3)]
    AfIm_cut_10 = AfIm[int(h/2):h, int(2*w/3):w]
    after = [AfIm, AfIm_cut_1, AfIm_cut_2, AfIm_cut_3, AfIm_cut_4, AfIm_cut_5, AfIm_cut_6, AfIm_cut_7, AfIm_cut_8, AfIm_cut_9, AfIm_cut_10]
    
    imglabel = raw_input("Whole image label? for transfer " + str(x) + ' (p or n)')
    while imglabel not in 'pn':
            imglabel = raw_input("Whole image label? p or n")

    for p in range(0,11):
        if imglabel == 'p':
            cv2.imshow('Before', (cv2.resize(before[p],(int(.4*w/2), int(.4*h/2)))))
            cv2.imshow('After', (cv2.resize(after[p],(int(.4*w/2), int(.4*h/2)))))
            label = raw_input("label?")
            fw.writerow([BeforeName+ '_' +str(p) +'.jpg', label])
        else:
            fw.writerow([BeforeName+ '_' +str(p) +'.jpg', "negative"])
            
                    
    

#print all_labels
'''
cv2.imshow('Before UR', (cv2.resize(BefIm_cut_1,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()
cv2.imshow('After UR', (cv2.resize(AfIm_cut_1,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()

cv2.imshow('Before UL', (cv2.resize(BefIm_cut_2,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()
cv2.imshow('After UL', (cv2.resize(AfIm_cut_2,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()

cv2.imshow('Bfore LR', (cv2.resize(BefIm_cut_3,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()
cv2.imshow('After LR', (cv2.resize(AfIm_cut_3,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()

cv2.imshow('Before LL', (cv2.resize(BefIm_cut_4,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()
cv2.imshow('After LL', (cv2.resize(AfIm_cut_4,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()

cv2.imshow('Before', (cv2.resize(BefIm,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()
cv2.imshow('After', (cv2.resize(AffIm,(int(.4*w/2), int(.4*h/2)))))
cv2.waitKey()
'''
f.close()
