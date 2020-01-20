import cv2
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(100,100,3)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# #model.add(Conv2D(32, (3, 3), activation='relu'))
# #model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # model.add(Dropout(0.5))
# model.add(Dense(73, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adam(0.00008),
#               metrics=['accuracy'])
# model.summary()
# model.load_weights('model_out.dms')
model= load_model('model_out.dms')
model.summary()

img = np.asarray([cv2.imread('test.jpg')])
img = img.astype('float32')
img /= 255
# test_image = image.load_img('test.jpg')
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
result = model.predict(img)

np.set_printoptions(precision=3, suppress=False)
print(result)
dir = {'CARAMEL': 0, 'LIQUORICE': 1, 'TROPICALPUNCH': 2, 'CRANBERRYANDAPPLE': 3, 'CHILLI': 4, 'GRAPE': 5, 'BUTTERSCOTCH': 6, 'BANNANASPLIT': 7, 'WILDCHERRY': 8, 'CANDYFLOSS': 9, 'PINKGRAPEFRUIT': 10, 'PINACOLADA': 11, 'WATERMELON': 12, 'TANGERINE': 13, 'STRAWBERRYSMOOTHIE': 14, 'STRAWBERRY': 15, 'SOURLEMON': 16, 'MARSHMALLOW': 17, 'COFFEE': 18, 'MANGO': 19, 'PEACHYPIE': 20, 'COLA': 21, 'MINTSORBET': 22, 'GRANNYSMITHAPPLE': 23, 'BLUEBERRYPIE': 24, 'LEMONANDLIME': 25, 'SOUTHSEAKIWI': 26, 'PEAR': 27, 'FRENCHVANILLA': 28, 'SOUTHSEASKIWI': 29, 'BANANASPLIT': 30}
for i,n in sorted(enumerate(result[0]), key=lambda x:x[1], reverse=True):
    if n>0:
        print(list(dir.keys())[i] + ' ' + str(n*100) + '%')
