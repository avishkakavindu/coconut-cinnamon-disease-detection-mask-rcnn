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
    NAME = "coconut_diseases_coco"
    NUM_CLASSES = 1 + 3
    STEPS_PER_EPOCH = 1
    USE_MINI_MASK = False
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class MaskRCNNModel:
    def __init__(self):
        ROOT_DIR = os.path.abspath("./")
        sys.path.append(ROOT_DIR)
        self.config = ModelConfig()
        self.model = MaskRCNN(mode='inference', model_dir='logs', config=self.config)
        trained_model_path = os.path.join(settings.BASE_DIR, 'api/utils/coconut_disease/logs', f'{self.config.NAME}\mask_rcnn_{self.config.NAME}.h5')
        # trained_model_path = os.path.join('logs', f'{self.config.NAME}\mask_rcnn_{self.config.NAME}.h5')
        self.model.load_weights(trained_model_path, by_name=True)

        self.class_names = ['BG', 'black_spot', 'brown_blight', 'tip_burn']

    def predict(self, img_path):
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
            display_instances(
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

    # Show detected objects in color and all others in B&W
    def color_splash(self, img, mask):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
        Returns result image.
        """
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 255
        # Copy color pixels from the original color image where mask is set
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, img, gray).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)
        return splash

    def detect_and_color_splash(self, model, image_path=None, video_path=None):
        assert image_path

        # Run model detection and generate the color splash effect
        # print("Running on {}".format(img))
        # Read image
        img = skimage.io.imread(image_path)

        global session
        global graph
        with graph.as_default():
            set_session(session)
            # Detect objects
            r = model.detect([img], verbose=1)[0]
        # Color splash
        splash = self.color_splash(img, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)

        print("Saved to ", file_name)

# Example Execution
# model = MaskRCNNModel()
# pred_to_text = model.predict('bb.jpeg')
# print(pred_to_text)