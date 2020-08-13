import pickle
from os import path, listdir, mkdir
from os.path import join
import sys
from keras.optimizers import SGD
# import splitfolders as sf   - a good library for splitting dataset to train/val/test
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.model_selection import train_test_split, cross_val_score
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf

labels = []
features = []
train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
train_path = 'origin'
save_path = 'saved_files'
fixed_size = tuple((200, 200))
bins = 8
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
home = sys.path[0]
epochs = 20
sessions = 5
model_name = 'CNN_model'
history_name = 'CNN_history'

def import_data():

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    # this is a generator that will read pictures found in
    # subfolers of 'data/origin', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        join(home, train_path),  # this is the target directory
        target_size=fixed_size,  # all images will be resized to fixed_size
        batch_size=batch_size,
        class_mode='sparse',
        subset='training')  # since we use categorical_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = train_datagen.flow_from_directory(
        join(home, train_path),
        target_size=fixed_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation')

    return train_generator, validation_generator


def train_model(x_train, y_train, x_test, y_test):
    # what is the image data format convention
    if K.image_data_format() == "channels_first":
        input_shape = (3, fixed_size[0], fixed_size[1])
    else:
        input_shape = (fixed_size[0], fixed_size[1], 3)

    # Building a CNN Model
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
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
    model.add(Dense(3, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # origin [sessions] models each [epochs] times
    max_acc = 0.0
    for i in range(sessions):
        # model training and evaluation
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=2)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        # save model if it performed better
        if test_acc > max_acc:
            max_acc = test_acc
            model.save(join(home, save_path, model_name))
            with open(join(home, save_path, history_name), 'wb') as file:
                pickle.dump(history.history, file)
        print("accuracy: ", test_acc, "\n Loss:", test_loss)


def plot_progress(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def predict(model, image):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=1)

    filenames = test_generator.filenames
    nb_samples = len(filenames)

    predict = model.predict_generator(test_generator, steps=nb_samples)


try:
    saved_files = listdir(join(home, save_path))
    files_exist = [True for f in train_labels if saved_files.count(f + '.npy')]
    if files_exist.count(False) != 0 or len(files_exist) == 0:
        save_files()
except OSError as e:
    mkdir(join(home, save_path))
    save_files()

# x_train, x_test, y_train, y_test = import_data()
# train_model(x_train, y_train, x_test, y_test)
# model = load_model(join(home, save_path, model_name))
# model
# history = pickle.load(open(join(home, save_path, history_name), "rb"))
# plot_progress(history)
seperate()
