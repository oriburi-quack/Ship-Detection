import os
# path to OriBuri.py
path = '/mnt/j/mine/school/geocoding/SAR_software/'
os.chdir(path)

import pickle
import OriBuri as ob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend
import keras
import matplotlib.pyplot as plt
from os import listdir
from alive_progress import alive_bar

#####################################################################
####                                                             ####
####                    load Sentinel1 data                      ####
####                                                             ####
#####################################################################

with open('/mnt/h/ShipDetection_DATA/S1A_ulsan_harbor.pkl', 'rb') as f:
    stack = pickle.load(f)

ob.imgshow_sq(stack, (0, 2), 'S1A_DATA')

#####################################################################
####                                                             ####
####                           FBR image                         ####
####                                                             ####
#####################################################################

'''
reference: 

Thibault Taillade, Laetitia Thirion-Lefevre and Régis Guinvarc’h, 2020.
Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based Change Detection,
MDPI, Remote Sens. 2020, 12(11), 1720; https://doi.org/10.3390/rs12111720
'''

fbr = ob.getFBR(stack, 1, 4.5)
ob.imgshow(fbr, (0,1), 'FBR Image')

#####################################################################
####                                                             ####
####                             CFAR                            ####
####                                                             ####
#####################################################################

ratio = stack[-1].copy()
ratio['Band'] = stack[-1]['Band'] / fbr
ratio['Product Name'] = 'Ratio Image'
ob.imgshow(ratio['Band'], (0,10), 'ratio img')

img = ob.cfar_db(ratio, 0.01, 51)
ob.imgshow(img['Band'], (0,1), 'ship')

#####################################################################
####                                                             ####
####                    load Deep learning data                  ####
####                                                             ####
#####################################################################

chip_data = np.load('/mnt/h/ShipDetection_DATA/ship_training_data.npy')
labels = np.load('/mnt/h/ShipDetection_DATA/ship_label_data.npy')

chip_data = chip_data/100
#####################################################################
####                                                             ####
####                       Model Selection                       ####
####                                                             ####
#####################################################################

'''
CG-CNN (CFAR Cuided - Convolutional Neural Network)

reference:

Shao, et al, 2023.
CFAR-guided Convolutional Neural Network for Large Scale Scene SAR Ship Detection,
IEEE, Radar Conference, DOI:10.1109/RADARCONF2351548.2023.10149747
'''

keras.backend.image_data_format()

numclass = 2
img_depth = 1
img_height = 50
img_width = 50

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size = (3,3), input_shape = (50,50,1), padding = 'SAME', activation = 'relu'),
    keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'SAME'),
    keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'SAME', activation = 'relu'),
    keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'SAME'),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation = 'relu'),
    keras.layers.Dense(numclass, activation = 'softmax'),
    keras.layers.Dropout(0.25)
])

train, test, train_label, test_label = train_test_split(chip_data[1:] ,labels[1:] ,test_size=0.2, random_state=30)

train = train.reshape((train.shape[0],50,50,1)).astype('float32')
test = test.reshape((test.shape[0],50,50,1)).astype('float32')
train_label = to_categorical(train_label, numclass)
test_label = to_categorical(test_label, numclass)

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics = ['accuracy'])
model.summary()


# Model Training
model.fit(train, train_label, batch_size=64, epochs=5, verbose=1)

loss, accuracy = model.evaluate(test, test_label, batch_size=64, verbose = 1)
print('Accuracy: {} %'.format(accuracy*100))

# save trained model
# from keras.models import load_model
# model.save('/mnt/h/ShipDetection_DATA/LeNet_ShipDetection.h5')
# model = load_model('/mnt/h/ShipDetection_DATA/LeNet_inshoreShipDetection.h5')

#####################################################################
####                                                             ####
####                             Test                            ####
####                                                             ####
#####################################################################

def getidx (product):
    idx_y = []
    idx_x = []
    
    b, c = np.shape(product['Band'])
    for i in range (0, b):
        for j in range (0, c):
            if product['Band'][i,j]:
                idx_y.append(i)
                idx_x.append(j)
                
    return idx_y, idx_x

y, x = getidx(img)

test = np.zeros((len(y),50,50))

for i in range (0,len(y)):
    a = y[i]
    b = x[i]
    
    test[i,:,:] = stack[-1]['Band'][a-25:a+25, b-25:b+25]

test=test.reshape(len(y),50,50,1)
    
pred = model.predict(test/100)


idx = []
for i in range (0,len(y)):
    if pred.argmax(1)[i] == 1:
        idx.append(i)
        
ans = np.zeros(np.shape(img['Band']))
for i in range (0, len(idx)):
    ans[y[idx[i]], x[idx[i]]] = 1


'''
mask = ob.makePolygon(stack[0])
ans = ans * mask[0]
'''

mask = np.load('/mnt/h/ShipDetection_DATA/landMasking_polygon.npy')
ans = ans*mask

ans_i = []
ans_j = []
a, b = np.shape(ans)
for i in range (0, a):
    for j in range (0, b):
        if ans[i, j] == 1:
            ans_i.append(i)
            ans_j.append(j)

#ans[ans == 0] =  np.nan
fig = plt.imshow(stack[-1]['Band'], cmap = 'gray')
fig.set_clim(0,2)
fig = plt.scatter(x, y, 1, c='r', alpha = 0.6, label='Initial CFAR Result')
fig = plt.scatter(ans_j, ans_i, 1, c='#00FFFF', alpha = 0.6, label='Deep Learnign Result')
fig.set_clim(-1,1)
plt.legend()
plt.title('Inshore Ship detection Result')
plt.show()

fig = plt.imshow(stack[-1]['Band'], cmap = 'gray')
fig.set_clim(0,2)
fig = plt.scatter(ans_j, ans_i, 1, c='#00FFFF', alpha = 0.6)
fig.set_clim(-1,1)
plt.title('Inshore Ship detection Result')
plt.show()
