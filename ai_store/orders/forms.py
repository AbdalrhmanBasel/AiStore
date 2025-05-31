from django import forms
from .models import Order

class BillingForm(forms.ModelForm):
    class Meta:
        model = Order
        fields = ['first_name', 'last_name', 'email', 'phone_number', 'address_line1', 'address_line2', 'city', 'postal_code', 'country']
        widgets = {
            'address_line1': forms.TextInput(attrs={'placeholder': 'Address line 1'}),
            'address_line2': forms.TextInput(attrs={'placeholder': 'Address line 2'}),
            'city': forms.TextInput(attrs={'placeholder': 'City'}),
            'postal_code': forms.TextInput(attrs={'placeholder': 'Postal Code', 'maxlength': '30'}),
            'country': forms.TextInput(attrs={'placeholder': 'Country'}),
            'phone_number': forms.TextInput(attrs={'maxlength': '30'}),
        }
