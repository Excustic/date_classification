#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, Efcom Solutions ltd."
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import json
import os
import random
import string
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from flask import Flask, request, flash, render_template
from os.path import join
from werkzeug.utils import redirect, secure_filename

home = sys.path[0]
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_IMAGES'] = join(home, 'saved_files', 'predict')
app.config['UPLOAD_MODELS'] = join(home, 'saved_files', 'models')
RESTAPI_URL = "https://xxxxx.com"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
save_path = 'saved_files'
train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
fixed_size = tuple((200, 200))
model_name = 'CNN_model.h5'


def load_model_config():
    """
    Loads newest model or the default
    """
    if os.path.isdir(app.config['UPLOAD_MODELS']):
        latest_folder = sorted(os.listdir(app.config['UPLOAD_MODELS']), key=os.path.getmtime)[-1]
        with load_model(join(app.config['UPLOAD_MODELS'], latest_folder, 'model')) as model:
            model.load_weights(join(app.config['UPLOAD_MODELS'], latest_folder, 'weights'))
            return model
    model = load_model(join(home, save_path, model_name))
    model.load_weights(join(save_path, 'weights_best.hdf5'))
    return model


def model_score(filepath, filename):
    """
    Imports a pre-trained model, feeds (filepath/filename) to the neural network and predicts class with confidence
    """
    img = Image.open(join(filepath, filename))
    img = img.resize(fixed_size)
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, fixed_size[0], fixed_size[1], 3)
    p = loaded_model.predict(img).tolist()[0]
    result = {'label': train_labels[p.index(max(p))], 'confidence': max(p)}
    with open(join(filepath, 'result.json'), 'w') as f:
        json.dump(result, f)
    return result


def allowed_file(filename, mode=0):
    """
    Validates the file type, only jpg and png are allowed
    """
    # mode 0 - picture upload, mode 1 - model and weights upload
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[0:len(ALLOWED_EXTENSIONS) - mode]


@app.route('/store', methods=['GET', 'POST'])
def store_model():
    """
    REST function, used for uploading an updated model
    """
    if request.method == 'POST':
        if 'model' not in request.files and 'weights' not in request.files:
            flash('Missing files')
            return redirect(request.url)
        model = request.files['model']
        weights = request.files['weights']
        # submit an empty part without filename
        if model.filename == '' or weights.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if model and weights and allowed_file(model.filename, mode=1) and allowed_file(weights.filename, mode=1):
            model_filename = secure_filename(file.filename)
            weights_filename = secure_filename(file.filename)
            filepath = join(app.config['UPLOAD_MODELS'], filename.split('.')[0]
                            , datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            try:
                if not os.path.isdir(app.config['UPLOAD_MODELS']):
                    os.mkdir(app.config['UPLOAD_MODELS'])
                os.mkdir(filepath)
                os.mkdir(join(filepath, 'model'))
                os.mkdir(join(filepath, 'weights'))
                model.save(join(filepath, 'model', model_filename))
                weights.save(join(filepath, 'weights', weights_filename))
            except Exception as e:
                print(e)
                return "Encountered Error"
        return "Uploaded Successfully"
    return render_template("store.html")


@app.route('/score', methods=['GET', 'POST'])
def score():
    """
    REST function, receives an image and predicts its class
    """
    if request.method == 'POST':
        res_json = None
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = join(app.config['UPLOAD_IMAGES'], filename.split('.')[0])
            try:
                if not os.path.isdir(app.config['UPLOAD_IMAGES']):
                    os.mkdir(app.config['UPLOAD_IMAGES'])
                # check if there's an actual duplicate or just same name
                if os.path.isdir(filepath):
                    # make a new folder if there's a file with the same name
                    rand_str = ''.join(random.choice(string.ascii_letters) for i in range(5))
                    while os.path.isdir(filepath + rand_str):
                        rand_str = ''.join(random.choice(string.ascii_letters) for i in range(5))
                    filepath = filepath + rand_str
                os.mkdir(filepath)
                file.save(join(filepath, filename))
                res_json = model_score(filepath, filename)
            except Exception as e:
                print(e)
        return res_json
    return render_template("index.html")


if __name__ == '__main__':
    loaded_model = load_model_config()
    # configurations for the usage gpu_tensorflow
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    app.run(host='0.0.0.0', port=5000)
