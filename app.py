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
from importlib import import_module
from os import listdir
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, flash, render_template, Response, stream_with_context
from os.path import join, isdir
from werkzeug.utils import redirect, secure_filename
from fast_predict import create_lite, fast_predict
import threading

app = Flask(__name__, static_url_path='/static')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
save_path = 'saved_files'
home = Path.cwd()
app.config['UPLOAD_IMAGES'] = join(home, 'static', 'images')
app.config['UPLOAD_MODELS'] = join(home, save_path, 'models')
app.config['UPLOAD_CONFIGS'] = join(home, 'configs')
app.jinja_env.add_extension('jinja2.ext.do')
ALLOWED_EXTENSIONS = {'IMAGES': ['png', 'jpg', 'jpeg', 'bmp'], 'MODELS': ['h5', 'hdf5'], 'CONFIGS':['py']}
model_name = 'CNN_model.h5'
models = {}
res_generator = None


def load_model_config():
    """
    Load all models
    """
    global models
    if os.path.isdir(app.config['UPLOAD_MODELS']):
        fruits = listdir(app.config['UPLOAD_MODELS'])
        for f in fruits:
            options = listdir(join(app.config['UPLOAD_MODELS'], f))
            model_name = None
            weights = None
            for o in options:
                path = join(app.config['UPLOAD_MODELS'], f, o)
                if 'model.h5' in o.split('_'):
                    model_name = path
                elif 'weights.hdf5' in o.split('_'):
                    weights = path
            model = load_model(model_name)
            model.load_weights(weights)
            model = create_lite(model)
            # To avoid os-based errors we will make sure backlashes are converted to forward slashes
            model_name = model_name.replace('\\', '/')
            model_name = model_name.split('/')[-1]
            models[model_name] = model
            print('loaded model at path', model_name)
    else: os.mkdir(app.config['UPLOAD_MODELS'])


def allowed_file(filename, mode):
    """
    Validates the file type, only extensions from ALLOWED_EXTENSIONS
    """
    # mode 0 - picture upload, mode 1 - model and weights upload
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[mode]


@app.route('/store', methods=['GET', 'POST'])
def store_model():
    """
    REST function, used for uploading an updated model
    """
    task_names = []
    for folder in listdir(app.config['UPLOAD_MODELS']):
        task_names.append(folder)
    if request.method == 'POST':
        if 'model' not in request.files and 'weights' not in request.files:
            flash('No selected Files')
            return redirect(request.url)
        model = request.files['model']
        weights = request.files['weights']
        config = request.files['config']
        bg_type = 'no_bg' if request.form.get('clear_bg') else 'with_bg'
        task_type = request.form.get('task-type')
        name = task_type + '_' + bg_type
        # submit an empty part without filename
        if model.filename == '' or weights.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if config and not allowed_file(config.filename, "CONFIGS") or \
                not model or not weights or not allowed_file(model.filename, mode='MODELS') \
                or not allowed_file(weights.filename, mode='MODELS'):
            flash('Invalid file')
            return redirect(request.url)
        try:
            if not isdir(app.config['UPLOAD_MODELS']):
                os.mkdir(app.config['UPLOAD_MODELS'])
            new_dir = join(app.config['UPLOAD_MODELS'], name.split('_')[0])
            if not isdir(new_dir):
                os.mkdir(new_dir)
            for file in listdir(new_dir):
                if name in file.replace('\\','/').split('/')[-1]:
                    os.remove(join(new_dir, file))
            model_path = join(new_dir, name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + 'model.h5')
            weights_path = join(new_dir, name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + 'weights.hdf5')
            model.save(model_path)
            weights.save(weights_path)
            config.save(join(app.config['UPLOAD_CONFIGS'], task_type + "_config.py"))
            threading.Thread(target=import_model, args=(model_path, weights_path, model_path.replace("\\","/").split('/')[-1])).start()
        except Exception as e:
                print(e)
                return "Encountered Error"

        flash("Uploaded Successfully")
        return render_template("store.html", task_names=task_names)
    return render_template("store.html", task_names=task_names)

def import_model(model_path, weights_path, name):
    global models
    new_model = load_model(model_path)
    new_model.load_weights(weights_path)
    new_model = create_lite(new_model)
    models[name] = new_model
    print("model ready")

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv

def single_score(model, files, task):
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
            res_json = fast_predict(model, filepath, filename, task.split('_')[0])
            final_path = filepath.replace("\\","/").split('/')[-1] + '/' + filename
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
    model_names = []
    model_labels = {}
    for folder in listdir(app.config['UPLOAD_MODELS']):
        for file in listdir(join(app.config['UPLOAD_MODELS'], folder)):
            if 'model.h5' in file.replace('\\', '/').split('_'):
                model_names.append(file)
        model_labels[folder] = ' | '.join(import_module('configs.' + folder.replace("\\","/")
                                                        .split('/')[-1].split('_')[0] + '_config').train_labels)
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
            if not allowed_file(file.filename, 'IMAGES'):
                flash('Invalid file type')
                return render_template('index.html')
                # load specific model for the task
        try:
            name = request.form.get('task-type')
            model = models[name]
            return Response(stream_with_context(stream_template("index.html", gen=single_score(model, files, name), model_names=model_names, labels=model_labels)))
        except KeyError:
            flash('This setting is not available')
            render_template('index.html')
        except Exception as e:
            flash('Something went wrong')
            print(e)
            render_template('index.html')
    return render_template("index.html", model_names=model_names, labels=model_labels)


if __name__ == '__main__':
    load_model_config()
    # configurations for the usage gpu_tensorflow
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    app.run(host='0.0.0.0', port=5000, debug=True)
