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
import glob
import os
import random
import shutil
import string
import sys
from os import listdir
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, flash, render_template, Response, stream_with_context
from os.path import join
from werkzeug.utils import redirect, secure_filename
from fast_predict import create_lite, fast_predict

app = Flask(__name__, static_url_path='/static')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
save_path = 'saved_files'
home = sys.path[0]
app.config['UPLOAD_IMAGES'] = join(home, 'static', 'images')
app.config['UPLOAD_MODELS'] = join(home, save_path, 'models')
app.jinja_env.add_extension('jinja2.ext.do')
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'h5', 'hdf5']
model_name = 'CNN_model.h5'
labels = {}
confidences = {}
filepaths = {}
res_generator = None


def load_model_config():
    """
    Loads newest model or the default
    """
    model = load_model(join(home, save_path, model_name))
    model.load_weights(join(save_path, 'weights_best.hdf5'))
    if os.path.isdir(app.config['UPLOAD_MODELS']):
        latest_folder = max(glob.glob(join(app.config['UPLOAD_MODELS'], '*')), key=os.path.getctime)
        try:
            model = load_model(join(app.config['UPLOAD_MODELS'], latest_folder, 'model',
                                    os.listdir(join(app.config['UPLOAD_MODELS'], latest_folder, 'model'))[0]))
            model.load_weights(join(app.config['UPLOAD_MODELS'], latest_folder, 'weights',
                                    listdir(join(app.config['UPLOAD_MODELS'], latest_folder, 'weights'))[0]))
            return model
        except Exception as e:
            print(e)
    return model


def allowed_file(filename, mode=0):
    """
    Validates the file type, only jpg and png are allowed
    """
    # mode 0 - picture upload, mode 1 - model and weights upload
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[(len(ALLOWED_EXTENSIONS) - 2)
                                                                    * mode:(len(ALLOWED_EXTENSIONS) - 2) + mode * 2]


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
        if not model or not weights or not allowed_file(model.filename, mode=1) or not allowed_file(weights.filename,
                                                                                                    mode=1):
            flash('Invalid file')
            return redirect(request.url)
        model_filename = secure_filename(model.filename)
        weights_filename = secure_filename(weights.filename)
        filepath = join(app.config['UPLOAD_MODELS'], model_filename.split('.')[0]
                        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv

def single_score(files):
    """
    Generates a score for each file
    :return generator object
    """
    for file in files:
        filename = secure_filename(file.filename)
        filepath = join(app.config['UPLOAD_IMAGES'], filename.split('.')[0])
        try:
            if not os.path.isdir(app.config['UPLOAD_IMAGES']):
                os.mkdir(app.config['UPLOAD_IMAGES'])
            # check if there's an actual duplicate or just same name
            if os.path.isdir(filepath):
                # make a new folder if there's a file with the same name
                rand_str = ''.join(random.choice(string.ascii_letters) for _ in range(5))
                while os.path.isdir(filepath + rand_str):
                    rand_str = ''.join(random.choice(string.ascii_letters) for _ in range(5))
                filepath = filepath + rand_str
            os.mkdir(filepath)
            file.save(join(filepath, filename))
            # fast_predict is a fast implementation of the basic predict method
            res_json = fast_predict(filepath, filename)
            final_path = filepath.split("\\")[-1] + '/' + filename
            filepath = final_path
            label = res_json["label"]
            confidence = res_json["confidence"]
            obj = {'confidence': "{:.2%}".format(confidence), 'label': label, 'filepath': filepath, 'filename': filename}
            yield obj
        except Exception as e:
            print(e)
            
@app.route('/score', methods=['GET', 'POST'])
def score():
    """
    REST function, receives an image and predicts its class
    """
    global filepaths
    global labels
    global confidences
    if request.method == 'POST':
        shutil.rmtree(app.config['UPLOAD_IMAGES'])
        os.mkdir(app.config['UPLOAD_IMAGES'])
        # check if the post request has the file part
        if 'file[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('file[]')
        for file in files:
            if not file or file.filename == '':
                flash('No selected file')
                return render_template('index.html')
            if not allowed_file(file.filename):
                flash('Invalid file type')
                return render_template('index.html')
        flash('Selected ' + str(len(files)) + ' file(s)')
        return Response(stream_with_context(stream_template("index.html", gen=single_score(files))))
    filepaths = {}
    labels = {}
    confidences = {}
    return render_template("index.html")


if __name__ == '__main__':
    loaded_model = load_model_config()
    create_lite(loaded_model)
    # configurations for the usage gpu_tensorflow
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    app.run(host='0.0.0.0', port=5000, debug=True)
