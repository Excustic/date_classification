import datetime
import multiprocessing
import pickle
from os import path, listdir, mkdir
from os.path import join, isdir
import sys

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
# import splitfolders as sf   - a good library for splitting dataset to train/val/test
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50

labels = []
features = []
train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
train_path = 'split\\train'
valid_path = 'split\\val'
test_path = 'split\\test'
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

def import_data():

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
        )  # since we use categorical_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = val_datagen.flow_from_directory(
        join(home, valid_path),
        target_size=fixed_size,
        batch_size=batch_size,
        class_mode='sparse',
        )

    return train_generator, validation_generator



def build_model():
    pretrained_model = InceptionV3(input_shape=(fixed_size[0], fixed_size[1], 3), weights='imagenet', include_top=False)
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
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=batch_size)
    model = build_model()
    # checkpoint
    filepath = join(save_path, "weights_best_smart.hdf5")
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

train_generator, validation_generator = import_data()
train_model(train_generator, validation_generator)
# model = load_model(join(home, save_path, model_name))
# history = pickle.load(open(join(home, save_path, history_name), "rb"))
# plot_progress(history)
# print(model.evaluate_generator(validation_generator, 250))
