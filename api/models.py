from django.db import models


class Cure(models.Model):
    # Coconut diseases
    BLACK_SPOT = 'black_spot'
    BROWN_BLIGHT = 'brown_blight'
    TIP_BURN = 'tip_burn'
    # Cinnamon diseases
    ROUGH_BARK = 'RoughBark'
    STRIP_CANCER = 'StripeCanker'

    CINNAMON = 'cinnamon'
    COCONUT = 'coconut'

    DISEASE_CHOICES = [
        (BLACK_SPOT, 'Black Spot'),
        (BROWN_BLIGHT, 'Brown Blight'),
        (TIP_BURN, 'Tip Burn'),

        (ROUGH_BARK, 'Rough Bark'),
        (STRIP_CANCER, 'Strip Cancer')
    ]

    PLANT_CHOICES = [
        (CINNAMON, 'Cinnamon'),
        (COCONUT, 'Coconut')
    ]

    disease = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    cure_description = models.TextField()
    plant = models.CharField(max_length=20, choices=PLANT_CHOICES, default=COCONUT)

    def __str__(self):
        return f'{self.get_plant_display()} - {self.get_disease_display()}'
