from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json


class ImageProcessor(ImageProcessorBase):

    # OPTIONAL: Use this method to preprocess images using the image objects
    #           they've been loaded into automatically.
    #           You can skip this and just perform the preprocessing after
    #           the input image has been convertet to a numpy array (see below).
    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):

            image = image.resize((256,256))
            # image = np.array(image)
            # TODO: implement preprocessing of PIL image objects
        elif isinstance(image, SimpleITK.Image):
            pass
            # TODO: implement preprocessing of SimpleITK image objects
        else:
            raise IOError("Image Type not supported for preprocessing.")
        return image


    def _preprocessAfterConversionToNumpy(self, npArr):
        # TODO: implement preprocessing of image after it was converted to a numpy array
        npArr = npArr.astype('float32')
        npArr /= 255.0
        npArr = npArr.reshape(1,256,256,1)
        return npArr


    def computeOutput(self, inferenceResults):
        # TODO: implement postprocessing of inference results
        return inferenceResults
