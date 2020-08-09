from os import path, listdir, mkdir
import sys
import cv2
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split, cross_val_score
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

labels = []
features = []
train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
train_path = 'Ready_For_Model_3Var'
save_path = 'saved_files'
fixed_size = tuple((200, 200))
bins = 8
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
home = sys.path[0]

def save_files():
    # loop over the training data sub-folders
    for training_name in train_labels:

        photos = []

        # join the training data path and each species training folder
        dir = path.join(home, train_path, training_name)

        # get the current training label
        current_label = training_name.split('_')[1]

        # loop over the images in each sub-folder
        for filename in listdir(dir):
            if filename.split('.')[1] != "jpg":
                continue
            # load image
            photo = load_img(path.join(dir, filename), target_size=fixed_size)
            # convert to numpy array
            photo = img_to_array(photo)/255.0
            # store
            photos.append(photo)

        # convert to a numpy arrays
        photos = np.asarray(photos)

        # save the reshaped photos
        np.save(path.join(home, save_path, training_name+'.npy'), photos)


try:
    saved_files = listdir(path.join(home, save_path))
    files_exist = [True for f in train_labels if saved_files.count(f+'.npy')]
    if files_exist.count(False) != 0 or len(files_exist) == 0:
        save_files()
except OSError as e:
    mkdir(path.join(home, save_path))
    save_files()

seed = 11
test_size = 0.10
# features = np.array(h5py.File(path.join(save_path, h5_data), 'r')
#                     .get('dataset_1'))
# labels = np.array(h5py.File(path.join(save_path, h5_labels), 'r')
#                     .get('dataset_1'))
x = np.array(tuple(np.load(path.join(home, save_path, train_labels[i]+'.npy')) for i in range(3)))
y = np.array(tuple(np.array([i] * len(x[i])) for i in range(3)))

X = np.vstack(x)
labels = np.hstack(y)
# split the training and testing data
(x_train, x_test, y_train, y_test) = train_test_split(X, labels,
                                                      test_size=test_size,
                                                      random_state=seed)


# what is the image data format convention
if K.image_data_format() == "channels_first":
    input_shape = (3, fixed_size[0], fixed_size[1])
else:
    input_shape = (fixed_size[0], fixed_size[1], 3)

# Building a CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.0003, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("accuracy: ", test_acc, "\n Loss:", test_loss)
