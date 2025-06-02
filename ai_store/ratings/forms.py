# ratings/forms.py
from django import forms
from .models import Rating

class RatingForm(forms.ModelForm):
    class Meta:
        model = Rating
        fields = ['rating']
        widgets = {
            'rating': forms.RadioSelect(choices=[(i, f'{i} ⭐') for i in range(1, 6)]),
        }
