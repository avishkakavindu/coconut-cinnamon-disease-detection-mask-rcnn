from django.urls import path

from .api_views import DetectCoconutDiseases, DetectCinnamonDiseases

urlpatterns = [
    path('coconut/predict', DetectCoconutDiseases.as_view(), name='coconut-disease'),
    path('cinnamon/predict', DetectCinnamonDiseases.as_view(), name='cinnamon-disease')
]