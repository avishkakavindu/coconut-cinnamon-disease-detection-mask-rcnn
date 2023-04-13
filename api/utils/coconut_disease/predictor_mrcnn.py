import os
import sys
import skimage
import matplotlib.pyplot as plt
import numpy as np
import datetime
from django.conf import settings
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()
set_session(session)

from .mrcnn.config import Config
from .mrcnn.model import MaskRCNN
from .mrcnn.visualize import display_instances


# Define the configuration for the model
class ModelConfig(Config):
    """
    Configuration for the Mask R-CNN model.
    """
    NAME = "coconut_diseases_coco"
    NUM_CLASSES = 1 + 3
    STEPS_PER_EPOCH = 1
    USE_MINI_MASK = False
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNModel:
    """
        Wrapper class for the Mask R-CNN model.
    """

    def __init__(self):
        """
        Initializes the Mask R-CNN model and loads the pre-trained weights.
        """
        ROOT_DIR = os.path.abspath("./")
        sys.path.append(ROOT_DIR)
        self.config = ModelConfig()
        self.model = MaskRCNN(mode='inference', model_dir='logs', config=self.config)
        trained_model_path = os.path.join(settings.BASE_DIR, 'api/utils/coconut_disease/logs', f'{self.config.NAME}\mask_rcnn_{self.config.NAME}.h5')
        # trained_model_path = os.path.join('logs', f'{self.config.NAME}\mask_rcnn_{self.config.NAME}.h5')
        self.model.load_weights(trained_model_path, by_name=True)

        self.class_names = ['BG', 'black_spot', 'brown_blight', 'tip_burn']

    def predict(self, img_path):
        """
        Runs object detection on the specified image and returns the predicted class names.

        Parameters:
        img_path (str): path to the input image

        Returns:
        pred_to_text (list): predicted class names of the detected objects
        """
        # Load the input image
        sample_img = skimage.io.imread(img_path)
        # plt.imshow(sample_img)

        global session
        global graph
        with graph.as_default():
            set_session(session)
            # Perform object detection on the input image
            detected = self.model.detect([sample_img])
            results = detected[0]

            # Display the results of the object detection
            display_inpstances(
                sample_img,
                results['rois'],
                results['masks'],
                results['class_ids'],
                self.class_names,
                results['scores'],
                "", # title
                (16, 16),   # figsize
                None,   # ax
                True,   # show_mask=
                True,   # show_mask_polygon
                True,   # show_bbox
                None,   # colors
                None,   # captions
                True,   #show_caption
                img_path,   # save the plot
                None,   # filter_classes
                None    # min_score
            )

        # Output the predicted class names of the detected objects
        predicted_class_ids = detected[0]['class_ids']
        pred_to_text = [self.class_names[class_id] for class_id in predicted_class_ids]
        return pred_to_text

# model = MaskRCNNModel()
# pred_to_text = model.predict('bb.jpeg')
# print(pred_to_text)