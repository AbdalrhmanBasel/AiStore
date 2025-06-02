# ratings/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from store.models import Product
from .models import Rating, Review
from .forms import RatingForm
from tracking.hooks import log_interaction  # ⭐ Add this

# ⭐ Add or update rating (1 per user-product)
@login_required
def add_rating(request, product_id):
    product = get_object_or_404(Product, id=product_id)

    rating, created = Rating.objects.get_or_create(user=request.user, product=product)

    if request.method == 'POST':
        form = RatingForm(request.POST, instance=rating)
        if form.is_valid():
            form.save()

            # ⭐ Log rating interaction
            log_interaction(user_id=request.user.id, product_id=product.id, interaction_type='rating')

            if created:
                messages.success(request, 'Ваша оценка была добавлена.')
            else:
                messages.success(request, 'Ваша оценка была обновлена.')

            return redirect(product.get_absolute_url())
    else:
        form = RatingForm(instance=rating)

    return render(request, 'ratings/add_rating.html', {
        'form': form,
        'product': product
    })

# 📝 Add review (many per user-product)
@login_required
def add_review(request, product_id):
    product = get_object_or_404(Product, id=product_id)

    if request.method == 'POST':
        comment = request.POST.get('comment')
        if comment:
            Review.objects.create(user=request.user, product=product, comment=comment)
            messages.success(request, 'Ваш отзыв был добавлен.')

    return redirect(product.get_absolute_url())
