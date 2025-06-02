# store/views.py

from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from django.db.models import Q, Avg
from .models import Product
from categories.models import Category
from tracking.hooks import log_interaction

# Import recommender utils
from recommender.utils import load_encoder, get_products_from_encoded
from recommender.gnn.inference import get_recommendations

def store(request, category_slug=None):
    products = Product.objects.filter(is_available=True, stock__gt=0)

    query = request.GET.get('q', '').strip()

    if category_slug:
        category = get_object_or_404(Category, slug=category_slug)
        products = products.filter(category=category)
    
    if query:
        products = products.filter(
            Q(product_name__icontains=query) |
            Q(description__icontains=query) |
            Q(category__category_name__icontains=query)
        ).distinct()

    paginator = Paginator(products, 6)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # 🎁 RECOMMENDATIONS
    recommended_products = []
    user_encoder, product_encoder = load_encoder()

    if request.user.is_authenticated and user_encoder and product_encoder:
        try:
            user_id_enc = user_encoder.transform([request.user.id])[0]
            recommended_product_ids = get_recommendations(user_id_enc=user_id_enc, top_n=5)
            recommended_products = get_products_from_encoded(recommended_product_ids, product_encoder)
            print(f"✅ Store page recommendations for user {request.user.id}: {[p.id for p in recommended_products]}")
        except Exception as e:
            print(f"❌ Failed to generate recommendations in store view: {e}")

    context = {
        'products': page_obj.object_list,
        'products_count': paginator.count,
        'category_slug': category_slug,
        'query': query,
        'page_obj': page_obj,
        'recommended_products': recommended_products,
    }

    return render(request, 'store/store.html', context)

def product_detail(request, category_slug, product_slug):
    product = get_object_or_404(Product, slug=product_slug, category__slug=category_slug)

    # ⭐ Track 'view' interaction — with ref tracking
    ref = request.GET.get('ref')
    if request.user.is_authenticated:
        interaction_type = 'view'
        if ref == 'recommend':
            interaction_type = 'recommend_click'
        log_interaction(user_id=request.user.id, product_id=product.id, interaction_type=interaction_type)

    # Ratings logic
    user_rating = None
    avg_rating = product.ratings.aggregate(avg=Avg('rating'))['avg']

    if request.user.is_authenticated:
        user_rating = product.ratings.filter(user=request.user).first()

    context = {
        'product': product,
        'user_rating': user_rating,
        'avg_rating': avg_rating,
        'rating_range': range(1, 6),
    }
    return render(request, 'store/product_detail.html', context)
