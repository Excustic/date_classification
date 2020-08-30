from os.path import join
from time import time
import tensorflow as tf
import numpy as np
from PIL import Image
from importlib import import_module


class LiteModel:

    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array(inp, dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0].tolist()


def fast_predict(lite_model, filepath, filename, task):
    """
    Imports a pre-trained model, feeds (filepath/filename) to the Lite neural network and predicts class with confidence
    This method is much faster than standard score, 50x factor!
    """
    # Start a stopper
    t0 = time()
    # Pillow library is used since we open a new file that wasn't in our test folder
    config = import_module('configs.'+task+'_config')
    fixed_size = config.fixed_size
    train_labels = config.train_labels
    img = Image.open(join(filepath, filename))
    img = img.resize(fixed_size)
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, fixed_size[0], fixed_size[1], 3)
    p = lite_model.predict_single(img)
    result = {'label': train_labels[p.index(max(p))], 'confidence': max(p)}
    # Print recorded time
    print("%.4f sec" % (time() - t0))
    return result


def create_lite(model):
    return LiteModel.from_keras_model(model)