from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from orders.models import Order
from .forms import ProfileForm
@login_required
def user_dashboard(request):
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'dashboard/dashboard.html', {'orders': orders})


@login_required
def edit_profile(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile was updated successfully.')
            return redirect('user_dashboard')
    else:
        form = ProfileForm(instance=request.user)

    return render(request, 'dashboard/edit_profile.html', {'form': form})
