from keras.models import *
import json
from processing import ImageProcessor
from modelhublib.model import ModelBase


class Model(ModelBase):

    def __init__(self):
        # load config file
        config = json.load(open("model/config.json"))
        # get the image processor
        self._imageProcessor = ImageProcessor(config)
        # load the DL model within keras (change thi if you are not using keras)
        self._model = load_model('model/model.h5')

    def infer(self, input):
        # load preprocessed input
        inputAsNpArr = self._imageProcessor.loadAndPreprocess(input)
        # Run inference with kreas (change this if you are not using keras)
        results = self._model.predict(inputAsNpArr)        # postprocess results into output
        output = self._imageProcessor.computeOutput(results)
        return output
        

