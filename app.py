#!/usr/bin/env python
__author__ = "Michael Kushnir"
__copyright__ = "Copyright 2020, OnPoint Project"
__credits__ = ["Michael Kushnir"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Kushnir"
__email__ = "michaelkushnir123233@gmail.com"
__status__ = "prototype"

import os
import random
import string
import sys
from threading import Thread
import cv2
import numpy as np
from flask import Flask, request, flash, render_template
from os.path import join
import date_classification as clf
from werkzeug.utils import redirect, secure_filename

home = sys.path[0]
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = join(home, 'saved_files', 'predict')
RESTAPI_URL = "https://xxxxx.com"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_similar(local_img, server_img):
    npimg = np.fromstring(server_img.read().decode('cp850'), np.uint8)
    local_img = cv2.imdecode(np.fromstring(local_img, np.uint8), cv2.IMREAD_COLOR)
    npimg = cv2.imdecode(np.fromstring(npimg, np.uint8), cv2.IMREAD_COLOR)
    # compare image dimensions (assumption 1)
    if local_img.size != server_img.size:
        return False

    err = np.sum((local_img.astype("float") - npimg.astype("float")) ** 2)
    err /= float(local_img.shape[0] * local_img.shape[1])

    return err == 0


@app.route('/score', methods=['GET', 'POST'])
def score():
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
            filepath = join(app.config['UPLOAD_FOLDER'], filename.split('.')[0])
            try:
                if not os.path.isdir(app.config['UPLOAD_FOLDER']):
                    os.mkdir(app.config['UPLOAD_FOLDER'])
                # check if there's an actual duplicate or just same name
                if os.path.isdir(filepath):
                    # make a new folder if there's a file with the same name
                    rand_str = ''.join(random.choice(string.ascii_letters) for i in range(5))
                    while os.path.isdir(filepath+rand_str):
                        rand_str = ''.join(random.choice(string.ascii_letters) for i in range(5))
                    filepath = filepath+rand_str
                os.mkdir(filepath)
                file.save(join(filepath, filename))
                res_json = clf.score(filepath, filename)
            except Exception as e:
                print(e)
        return res_json
    return render_template("index.html")


if __name__ == '__main__':
    t = Thread(target=app.run, args=())
    t.start()
