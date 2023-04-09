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

from .models import Cure


model = predictor_mrcnn.MaskRCNNModel()


class DetectCoconutDiseases(APIView):
    """ Handles the logic for coconut disease prediction """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['image']

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
                cures[disease] = Cure.objects.get(disease=disease).cure_description
            except Cure.DoesNotExist:
                cures[disease] = 'Cure not found in database'

        context = {
            'image_url': image_url,
            'predictions': pred_to_text,
            'cures': cures
        }

        return Response(context)
