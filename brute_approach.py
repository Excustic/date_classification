import pickle
import sys
import datetime
from os.path import join

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
import splitfolders as sf
""""- a good library for splitting dataset to train/val/test"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.utils.vis_utils import plot_model

labels = []
features = []
train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
train_path = 'split\\train'
valid_path = 'split\\val'
test_path = 'split\\test'
save_path = 'saved_files'
origin_path = 'Ready_For_Model_3Var'
fixed_size = tuple((200, 200))
bins = 8
home = sys.path[0]
epochs = 20
sessions = 5
model_name = 'CNN_model'
history_name = 'CNN_history'
batch_size = 64

# configurations for the usage gpu_tensorflow
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def import_data():

    # this is the augmentation configuration we will use for training
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        )
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # this is the augmentation configuration we will use for training
    # this is a generator that will read pictures found in
    # subfolers of 'data/origin', and indefinitely generate
    # batches of augmented image data
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


def train_model(train_generator, validation_generator):
    # what is the image data format convention
    if K.image_data_format() == "channels_first":
        input_shape = (3, fixed_size[0], fixed_size[1])
    else:
        input_shape = (fixed_size[0], fixed_size[1], 3)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=False,
        class_mode='sparse',
        batch_size=batch_size)

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
    opt = Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    filepath = "weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=2, restore_best_weights=True)
    log_dir = join(home, save_path, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [early_stopping, checkpoint, tensorboard_callback]
    # origin [sessions] models each [epochs] times
    max_acc = 0.0
    for i in range(sessions):
        # model training and evaluation
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=20,
            validation_data=validation_generator,
            validation_steps=len(validation_generator)
            , verbose=2, callbacks=callbacks_list)
        test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples)
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


def calc_activations(model):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle=True,
        class_mode='sparse',
        batch_size=1)

    x, y = test_generator.next()
    activations = activation_model.predict(x)

    return activations, y


def display_activation(model, activations, y, col_size, row_size, act_index):
    activation = activations[act_index]
    activation_index = 0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
    for row in range(0, row_size):
        for col in range(0, col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='autumn')
            activation_index += 1
    fig.tight_layout(pad=1.6)
    fig.suptitle(train_labels[int(y[0])]+", Layer "+str(model.layers[act_index].name))
    plt.show()

train_gen, val_gen = import_data()
train_model(train_gen, val_gen)
model = load_model(join(home, save_path, model_name))
plot_model(model, to_file=join(home, save_path, 'model.png'))
while str(input()) != 'q':
    activations, y = calc_activations(model)
    layer_num = int(input())
    while layer_num != -1:
        try:
            display_activation(model, activations, y, 8, 4, layer_num)
        except Exception as e:
            print("failed - " + str(e))
        layer_num = int(input())

# model
# history = pickle.load(open(join(home, save_path, history_name), "rb"))
# plot_progress(history)
