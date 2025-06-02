from django.shortcuts import render
from store.models import Product
from recommender.utils import load_encoder, get_products_from_encoded
from recommender.gnn.inference import get_recommendations

def home(request):
    popular_products = Product.objects.filter(is_available=True, stock__gt=0).order_by('-popularity_score')[:8]

    recommended_products = []
    user_encoder, product_encoder = load_encoder()

    if request.user.is_authenticated and user_encoder and product_encoder:
        try:
            user_id_enc = user_encoder.transform([request.user.id])[0]
            recommended_product_ids = get_recommendations(user_id_enc=user_id_enc, top_n=8)
            recommended_products = get_products_from_encoded(recommended_product_ids, product_encoder)
            print(f"✅ Home page recommendations for user {request.user.id}: {[p.id for p in recommended_products]}")
        except Exception as e:
            print(f"❌ Failed to generate recommendations on home page: {e}")

    context = {
        'products': popular_products,
        'recommended_products': recommended_products,
    }
    return render(request, "home.html", context)
