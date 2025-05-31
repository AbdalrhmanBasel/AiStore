from django import forms
from .models import Account

from django.shortcuts import render, redirect
from django.contrib import messages


class RegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, label="Password")
    confirm_password = forms.CharField(widget=forms.PasswordInput, label="Confirm Password")

    class Meta:
        model = Account
        fields = ['first_name', 'last_name', 'email', 'phone_number']  # no username here

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")

        if password and confirm_password and password != confirm_password:
            self.add_error('confirm_password', "Passwords do not match")

        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        # Auto-generate username from email
        user.username = self.cleaned_data['email'].split('@')[0]
        user.set_password(self.cleaned_data["password"])
        if commit:
            user.save()
        return user




class ForgotPasswordForm(forms.Form):
    email = forms.EmailField(label='Email', max_length=254)

