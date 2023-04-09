from django.urls import path

from .api_views import DetectCoconutDiseases

urlpatterns = [
    path('coconut/predict', DetectCoconutDiseases.as_view(), name='coconut-disease')
]