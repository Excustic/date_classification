#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, Efcom Solutions ltd."
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import multiprocessing
import pickle
import sys
import datetime
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
import splitfolders as sf   # a good library for splitting dataset to train/val/test
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
train_path = 'split2\\train'
valid_path = 'split2\\val'
test_path = 'split2\\test'
save_path = 'saved_files'
fixed_size = tuple((200, 200))
home = sys.path[0]
epochs = 100
sessions = 1
model_name = 'CNN_model2.h5'
history_name = 'CNN_history2'
weights_path = "weights_best2.hdf5"
batch_size = 32     # larger size might not work on some machines
# configurations for the usage gpu_tensorflow
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def import_data():
    """
    In this module we use a technique of Image Augmentation called Image Data Generators,
    this function configures them
    """
    # this is the augmentation configuration we will use for training
    # you can tinker with values to avoid over-fitting or under-fitting; I found these values to do well
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # rescale pixel values from 0-255 to 0-1 so the data would be normalized
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # sub-folders and indefinitely generate batches of augmented image data
    train_generator = datagen.flow_from_directory(
        join(home, train_path),  # this is the target directory
        target_size=fixed_size,  # all images will be resized to fixed_size
        batch_size=batch_size,
        class_mode='sparse',
    )  # since we use sparse_categorical_crossentropy loss, we need sparse labels

    # this is a similar generator, for validation data
    validation_generator = val_datagen.flow_from_directory(
        join(home, valid_path),
        target_size=fixed_size,
        batch_size=batch_size,
        class_mode='sparse',
    )

    return train_generator, validation_generator


def train_model(train_generator, validation_generator):
    """
    Trains the model, requires train/val generators.
    A model with best accuracy will be stored as a file separately in the saved_files folder
    """
    # what is the image data format convention
    if K.image_data_format() == "channels_first":
        input_shape = (3, fixed_size[0], fixed_size[1])
    else:
        input_shape = (fixed_size[0], fixed_size[1], 3)

    # we build a test generator to benchmark the model on unseen data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=batch_size)

    # Building a CNN Model
    model = Sequential()
    model.add(
        Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    # compile model
    opt = Adam(learning_rate=.0004 * (batch_size // 32))
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(join(save_path, weights_path), monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=epochs // 5, verbose=1,
                                   restore_best_weights=True)
    log_dir = join(home, save_path, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [early_stopping, checkpoint, tensorboard_callback]
    # train [sessions] models, each [epochs] times
    max_acc = 0.0
    for i in range(sessions):
        # model training and evaluation
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
            , verbose=2, callbacks=callbacks_list, workers=multiprocessing.cpu_count(),
            use_multiprocessing=False)
        model.load_weights(join(save_path, weights_path))
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        # save model if it performed better
        if test_acc > max_acc:
            max_acc = test_acc
            model.save(join(home, save_path, model_name))
            with open(join(home, save_path, history_name), 'wb') as file:
                pickle.dump(history.history, file)
        print("accuracy: ", test_acc, "\n Loss:", test_loss)


def plot_progress(history):
    """
    ***DEPRECATED - Tensorboard is a better implementation***
    Uses history file of model to plot metrics
    """
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


def calc_activations(model, index):
    """
    Calculates activations for a single image and outputs the calculations and the filename of the image
    Used for display_activation
    """
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    # Import a single image
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=fixed_size,
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=1)
    # get image from generator using index - we use index to retrieve the filename
    x, _ = test_generator._get_batches_of_transformed_samples([index])
    filename = test_generator.filenames[index]
    activations = activation_model.predict(x)

    return activations, filename


def display_activation(model, activations, name, col_size, row_size, act_index):
    """
    Plots activations of an image fed to the model
    Useful for visualizing features the model is picking
    Used for visualize
    """
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='autumn')
            activation_index += 1
    fig.tight_layout(pad=1.6)
    fig.suptitle(name + ", Layer " + str(model.layers[act_index].name))
    plt.show()


def test_log(model):
    """
    A verbose log of test evaluation of the model
    """
    # Import the test data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=1)
    # Get the simple test
    print(model.evaluate(test_generator, steps=len(test_generator)))
    # Detailed test
    PR = ([], [], [])
    for i in range(test_generator.samples):
        x, y = test_generator._get_batches_of_transformed_samples([i])
        filepath = test_generator.filepaths[i]
        p = model.predict(x, ).tolist()[0]
        PR[int(y[0])].append(int(y[0]) == p.index(max(p)))
        print("prediction - ", train_labels[p.index(max(p))], " | real - ", train_labels[int(y[0])], "| confidence - ", max(p),
              "| f:", filepath)
    for i in range(3):
        print(train_labels[i], ": ", PR[i].count(True), "/", len(PR[i]), "correct - ",
              (PR[i].count(True) / len(PR[i]) * 100),
              "accuracy")


def visualize(model):
    """
    An interactive function which plots the features that layers of the model picked up
    To use, simply press enter to get a new image; enter a number from 0-6 to see what layer at that index is picking up
    enter -1 to advance to another image, finally press q after -1 to exit. enter -2 to see another class
    """
    index = 0
    current_label = train_labels[0]
    while str(input()) != 'q':
        activations, name = calc_activations(model, index)
        layer_num = int(input())
        while layer_num != -1:
            if layer_num == -2:
                tmp = current_label
                while current_label != train_labels[train_labels.index(tmp)+1]:
                    index += 20
                    _, name = calc_activations(model, index)
                    if name.split('\\')[0] == train_labels[train_labels.index(current_label)+1]:
                        current_label = train_labels[train_labels.index(current_label) + 1]
                break
            try:
                display_activation(model, activations, name, 8, 4, layer_num)
            except Exception as e:
                print("failed - " + str(e))
            layer_num = int(input())
        index += 1


def score(filepath, filename, model):
    """
    Imports a pre-trained model, feeds (filepath/filename) to the neural network and predicts class with confidence
    """
    # Pillow library is used since we open a new file that wasn't in our test folder
    img = Image.open(join(filepath, filename))
    img = img.resize(fixed_size)
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, fixed_size[0], fixed_size[1], 3)
    p = model.predict(img).tolist()[0]
    print(p)
    result = {'label': train_labels[p.index(max(p))], 'confidence': max(p)}
    return result