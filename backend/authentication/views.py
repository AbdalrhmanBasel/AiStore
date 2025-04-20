from django.shortcuts import render, HttpResponseRedirect
from django.contrib import messages
from django.urls import reverse

from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

from .forms import UserSignUpForm, ProfileForm
from .models import Profile

def user_signup(request):
    """
    TODO: The Goal is to enable user to signup
    via api post request.

    Problem: We want the user file the forms on the frontend
    and than send the form's data to the backend through POST
    request. 

    After that, the frontend recieves notification of user's creation
    success and redirect the user to login.

    Now here in the function, user is rendered to django templates.
    No APIs involved. We need to fix this and ensure that user can
    sign up and in from the frontend.
    """
    form = UserSignUpForm()

    if request.method == 'POST':
        form = UserSignUpForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, "User account has been created successfully.")
            return HttpResponseRedirect(reverse('App_Login:login'))
        
    return render(request, 'App_Login/sign_up.html', context={'form':form})


def user_login(request):
    """
    TODO: Enable user to login from frontend to homepage succesfuly.

    Problem: As you see in the code, the user to login doesn't enter
    his email but username. We need to ensure that this username refers
    to the user's email and not just his username.

    Additionally, similar to the `user_signup` function, the function 
    renders the user to django templates. We want to ensure that 
    user login from frontend, sends POST request to backend, the
    backend authenticates the process, and than the backend respond 
    to frontend end. 
    
    The frontend will authenticate the user without
    redirecting the user anywhere. Let his shopping journey 
    gets uninterrupted.
    """
    form = AuthenticationForm()

    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)

        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')

            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)
                return HttpResponseRedirect(reverse('App_Shop:home'))

    return render(request, 'App_Login/login.html', context={'form':form})

@login_required
def user_logout(request):
    """
    TODO: The Goal is to make the user logout from the frontend

    Problem: After user logs out, the user shall be redirected to
    the frontend's homepage.

    Here, the user is redirected to a django template, so fix that.
    """
    logout(request)
    messages.warning(request, "You have been logged out.")
    return HttpResponseRedirect(reverse('App_Shop:home'))

@login_required
def user_profile(request):
    """
    TODO: The Goal is to make the user update his profile's data
    from the frontend succesfully.

    Problem: User have to submit the form from the frontend to the backend.
    After that, the backend using this function will update the data for the 
    user.

    Notice that at this moment, the user is redirected to a django template.
    You have to update that so backend can only render apis.
    """
    profile = Profile.objects.get(user=request.user)
    form = ProfileForm

    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile)

        if form.is_valid():
            form.save()
            messages.success(request, "Profile has been updated successfully")
            form = ProfileForm(instance=profile)

    return render(request, 'App_Login/change_profile.html', context={'form':form})


def user_restore_password(request):
    """
    TODO: Build a function that enables the user to restore his password 
    through his email.

    The user will submit his restore password form on the frontend. After
    that, the user will recieve an email on his inbox with link to set new
    password.
    """
    pass