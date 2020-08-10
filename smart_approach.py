import pickle
from os import path, listdir, mkdir
from os.path import join, isdir
import sys

from keras.callbacks import ModelCheckpoint
from keras import applications, Model, Input
from keras.losses import BinaryCrossentropy
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
save_path = 'saved_files'
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
    # # what is the image data format convention
    # if K.image_data_format() == "channels_first":
    #     input_shape = (3, fixed_size[0], fixed_size[1])
    # else:
    #     input_shape = (fixed_size[0], fixed_size[1], 3)
    # # Create base model
    # base_model = keras.applications.Xception(
    #     weights='imagenet',
    #     input_shape=(150, 150, 3),
    #     include_top=False)
    # # Freeze base model
    # base_model.trainable = False
    #
    # # Create new model on top.
    # inputs = Input(shape=(150, 150, 3))
    # x = base_model(inputs, training=False)
    # x = GlobalAveragePooling2D()(x)
    # outputs = Dense(1)(x)
    # model = Model(inputs, outputs)
    #
    # loss_fn = BinaryCrossentropy(from_logits=True)
    # optimizer = Adam()
    #
    # # # Iterate over the batches of a dataset.
    # # for inputs, targets in new_dataset:
    # #     # Open a GradientTape.
    # #     with tf.GradientTape() as tape:
    # #         # Forward pass.
    # #         predictions = model(inputs)
    # #         # Compute the loss value for this batch.
    # #         loss_value = loss_fn(targets, predictions)
    #
    #     # # Get gradients of loss wrt the *trainable* weights.
    #     # gradients = tape.gradient(loss_value, model.trainable_weights)
    #     # # Update the weights of the model.
    #     # optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    #
    # return model


def train_model(train_generator, validation_generator):

    model = build_model()
    # checkpoint
    filepath = "weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # train [sessions] models each [epochs] times
    max_acc = 0.0
    for i in range(sessions):
        # model training and evaluation
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800 // batch_size
            , verbose=2, callbacks=callbacks_list)
        test_loss, test_acc = model.evaluate_generator(validation_generator, 500)
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
# train_model(train_generator, validation_generator)
# model = load_model(join(home, save_path, model_name))
# history = pickle.load(open(join(home, save_path, history_name), "rb"))
# plot_progress(history)
model = build_model()
model.load_weights("weights_best.hdf5")
print(model.evaluate_generator(validation_generator, 250))