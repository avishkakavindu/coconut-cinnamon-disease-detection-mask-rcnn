from django.contrib import admin

from .models import Cure


@admin.register(Cure)
class CureAdmin(admin.ModelAdmin):
    class Meta:
        model = Cure

