#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, Efcom Solutions ltd."
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import datetime
import multiprocessing
from os.path import join
import sys
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
# import splitfolders as sf   - a good library for splitting dataset to train/val/test
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from date_classification import import_data, plot_progress

labels = []
features = []
train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
train_path = 'split2\\train'
valid_path = 'split2\\val'
test_path = 'split2\\test'
save_path = 'saved_files'
fixed_size = tuple((200, 200))
bins = 8
home = sys.path[0]
epochs = 100
sessions = 1
model_name = 'InceptionV3_model.h5'
history_name = 'InceptionV3_history'
batch_size = 32

# configurations for the usage gpu_tensorflow
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def build_model():
    """
    This module uses the notion of Transfer-Learning
    I found VGG16 model to perform the best and nearly as good as mine
    This function configures our model, freezes the VGG 16 and adds a small module on top of it
    """
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
    opt = Adam(learning_rate=.0003)
    transfer_learning_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return transfer_learning_model


def train_model(train_generator, validation_generator):
    """
    Trains the model, requires train/val generators.
    A model with best accuracy will be stored as a file separately in the saved_files folder
    """
    # we build a test generator to benchmark the model on unseen data
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=batch_size)
    model = build_model()
    filepath = join(save_path, "weights_best_smart2.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, restore_best_weights=True)
    log_dir = join(home, save_path, 'logs', 'fit_smart', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [early_stopping, checkpoint, tensorboard_callback]
    # origin [sessions] models each [epochs] times
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
        model.load_weights(join(filepath))
        test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
        print("accuracy: ", test_acc, "\n Loss:", test_loss)
        K.clear_session()

# TODO: edit and a JSON too
def predict(model, image):
    p = model.model_score(image)
    print(p)


train_generator, validation_generator = import_data()
train_model(train_generator, validation_generator)
# model = load_model(join(home, save_path, model_name))
# history = pickle.load(open(join(home, save_path, history_name), "rb"))
# plot_progress(history)
# print(model.evaluate_generator(validation_generator, 250))
