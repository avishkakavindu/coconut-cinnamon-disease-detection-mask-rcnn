from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from datetime import datetime
import skimage.io
from rest_framework import status
import os
import sys

from .utils.coconut_disease import predictor_mrcnn
from .utils.cinnamon_disease import predictor

from .models import Cure

model = predictor_mrcnn.MaskRCNNModel()


class DetectCoconutDiseases(APIView):
    """ Handles the logic for coconut disease prediction """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            file_obj = request.FILES['image']
        except:
            return Response({'error': 'something is wrong with the uploaded file'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded image to a temporary file
        temp_image_file = default_storage.save('temp_image.jpg', ContentFile(file_obj.read()))

        img_path = os.path.join(settings.MEDIA_ROOT, temp_image_file)

        # Run object detection and generate the color splash effect sav save image
        pred_to_text = set(model.predict(img_path))

        # Construct the URL for the output image
        image_url = request.build_absolute_uri(settings.MEDIA_URL + temp_image_file)

        cures = {}
        for disease in pred_to_text:
            try:
                cures[disease] = Cure.objects.get(disease=disease, plant=Cure.COCONUT).cure_description
            except Cure.DoesNotExist:
                cures[disease] = 'Cure not found in database'

        context = {
            'image_url': image_url,
            'predictions': pred_to_text,
            'cures': cures
        }

        return Response(context)


class DetectCinnamonDiseases(APIView):
    """ Handles the logic for cinnamon disease prediction """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            file_obj = request.FILES['image']
        except:
            return Response({'error': 'something is wrong with the uploaded file'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the uploaded image to a temporary file
        temp_image_file = default_storage.save('temp_image.jpg', ContentFile(file_obj.read()))

        img_path = os.path.join(settings.MEDIA_ROOT, temp_image_file)

        # Run object detection and generate the color splash effect sav save image
        preds = predictor.get_cinnamon_predictions(img_path)

        cures = {}
        for disease in preds.keys():
            try:
                cures[disease] = Cure.objects.get(disease=disease, plant=Cure.CINNAMON).cure_description
            except Cure.DoesNotExist:
                cures[disease] = 'Cure not found in database'

        context = {
            'predictions': preds,
            'cures': cures
        }

        return Response(context)
