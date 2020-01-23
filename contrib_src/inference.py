from keras.models import *
from keras import backend as K
import json
from processing import ImageProcessor
from modelhublib.model import ModelBase
import tensorflow as tf


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model within keras (change thi if you are not using keras)

        self._session = tf.Session()
        K.set_session(self._session)
        self._model = load_model('model/model.h5')
        self._model._make_predict_function()

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)

        with self._session.as_default():
            with self._session.graph.as_default():
                # Run inference with kreas (change this if you are not using keras)
                results = self._model.predict(inputAsNpArr)
        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output
