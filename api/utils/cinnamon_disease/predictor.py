import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model

CLASSES = ['RoughBark', 'StripeCanker']


def get_processed_input_img(image_path, size=224):
    test_img = cv2.imread(image_path)
    test_img = cv2.resize(test_img, dsize=(size, size), interpolation=cv2.INTER_AREA)

    test_img = test_img.reshape((1, size, size, 3)).astype(np.float32)

    return test_img / 225


def get_cinnamon_predictions(img_path):
    # loading model
    loaded_model = load_model('api/utils/cinnamon_disease/saved_models/diseases_check_trained_custom_model.h5')

    processed_img = get_processed_input_img(img_path)
    pred = loaded_model.predict(processed_img)

    # inversely sorted array with indexes
    best_idx = (-pred).argsort()[0]

    pred_dict = {CLASSES[i]: pred[0][i] for i in best_idx}

    return pred_dict

# example usage
# img_path = './IMG_1563.JPG'
# preds = get_cinnamon_predictions(img_path)
# print(preds)
