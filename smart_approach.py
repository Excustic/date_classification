import pickle
from os import path, listdir, mkdir
from os.path import join, isdir
import sys

from keras.applications import VGG16, InceptionV3, Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import applications, Model, Input
from keras.losses import BinaryCrossentropy, sparse_categorical_crossentropy
from keras.metrics import Accuracy, SparseCategoricalCrossentropy
from keras.optimizers import SGD, Adam
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
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
train_path = 'Ready_For_Model_3Var'
save_path = '../../backup/saved_files'
fixed_size = tuple((200, 200))
home = sys.path[0]
epochs = 20
sessions = 5
model_name = 'CNN_model'
history_name = 'CNN_history'
train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 10
batch_size = 32

# configurations for the usage gpu_tensorflow
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def save_files():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # loop over the training data sub-folders
    for training_name in train_labels:

        photos = []

        # join the training data path and each species training folder
        dir = join(home, train_path, training_name)

        # get the current training label
        current_label = training_name.split('_')[1]

        # loop over the images in each sub-folder
        for filename in listdir(dir):
            # avoid non jpg files
            if filename.split('.')[1] != "jpg":
                continue
            # load image
            photo = load_img(join(dir, filename), target_size=fixed_size)
            # convert to numpy array
            photo = img_to_array(photo) / 255.0
            # store
            photos.append(photo)
        if not isdir(join(home, save_path, 'preview')):
            mkdir(join(home, save_path, 'preview'))
        photos = np.asarray(photos)
        i = 0
        for _ in datagen.flow(photos, batch_size=len(photos),
                              save_to_dir=join(home, save_path, 'preview'),
                              save_prefix=training_name, save_format='jpeg'):
            i += 1
            if i > 5:
                break  # otherwise the generator would loop indefinitely


def import_data():
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
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


def build_model():
    pass
    pretrained_model = VGG16(input_shape=(fixed_size[0], fixed_size[1], 3), weights='imagenet', include_top=False)
    # We will not train the layers imported.
    for layer in pretrained_model.layers:
        layer.trainable = False
    transfer_learning_model = Sequential()
    transfer_learning_model.add(pretrained_model)
    transfer_learning_model.add(Flatten())
    transfer_learning_model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    transfer_learning_model.add(Dropout(0.5))
    transfer_learning_model.add(Dense(3, activation='softmax'))
    transfer_learning_model.summary()

    return transfer_learning_model


def train_model(train_generator, validation_generator):
    model = build_model()
    # checkpoint
    filepath = "weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
    callbacks_list = [early_stopping, checkpoint]
    optimizer = Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # train [sessions] models each [epochs] times
    max_acc = 0.0
    for i in range(sessions):
        # model training and evaluation
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator) // batch_size,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=len(validation_generator) // batch_size
            , verbose=2, callbacks=callbacks_list)
        test_loss, test_acc = model.evaluate_generator(validation_generator, 500)
        # save model if it performed better
        # if test_acc > max_acc:
        #     max_acc = test_acc
        #     model.save(join(home, save_path, model_name))
        #     with open(join(home, save_path, history_name), 'wb') as file:
        #         pickle.dump(history.history, file)
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
    p = model.predict(image)
    print(p)


try:
    saved_files = listdir(join(home, save_path))
    files_exist = [True for f in train_labels if saved_files.count(f + '.npy')]
    if files_exist.count(False) != 0 or len(files_exist) == 0:
        save_files()
except OSError as e:
    mkdir(join(home, save_path))
    save_files()

train_generator, validation_generator = import_data()
train_model(train_generator, validation_generator)
# model = load_model(join(home, save_path, model_name))
# history = pickle.load(open(join(home, save_path, history_name), "rb"))
# plot_progress(history)
# print(model.evaluate_generator(validation_generator, 250))
