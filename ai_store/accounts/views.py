from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import RegistrationForm, ForgotPasswordForm

from orders.models import Order
from recommender.client import get_recommendations
from recommender.utils import load_encoder, get_products_from_encoded
from store.models import Product

# --- Register ---
def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, "Registration successful. Please log in.")
            return redirect('login')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = RegistrationForm()

    return render(request, 'accounts/register.html', {'form': form})

# --- Login ---
def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, email=email, password=password)

        if user is not None:
            auth_login(request, user)
            next_url = request.GET.get('next') or 'home'
            return redirect(next_url)
        else:
            messages.error(request, "Invalid email or password.")
            return redirect('login')

    return render(request, 'accounts/login.html')

# --- Logout ---
@login_required
def logout(request):
    auth_logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect('login')

# --- Forgot Password ---
def forgot_password(request):
    if request.method == 'POST':
        form = ForgotPasswordForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            # TODO: Send reset email
            messages.success(request, "If an account with this email exists, instructions have been sent.")
            return redirect('login')
        else:
            messages.error(request, "Please enter a valid email.")
    else:
        form = ForgotPasswordForm()
    return render(request, 'accounts/forgot_password.html', {'form': form})

# --- User Dashboard with Recommendations ---
@login_required
def user_dashboard(request):
    orders = Order.objects.filter(user=request.user).order_by('-created_at')

    # Load encoders
    user_encoder, product_encoder = load_encoder()

    # Encode user
    try:
        user_id_enc = user_encoder.transform([request.user.id])[0]

        # Get recommended product IDs (encoded)
        recommended_product_ids = get_recommendations(user_id_enc=user_id_enc, top_n=5)

        # Map back to Product objects
        recommendations = get_products_from_encoded(recommended_product_ids, product_encoder)

    except Exception as e:
        print(f"❌ Recommendation failed: {e}")
        recommendations = []

    # Render template
    return render(request, 'accounts/dashboard.html', {
        'orders': orders,
        'recommendations': recommendations
    })
