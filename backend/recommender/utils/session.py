# recommender/utils/session.py
from django.core.cache import cache
from django.db.models import Count
from store.models import Product

def get_session_recommendations(session_key, k=5):
    session_recs = cache.get(f"session_recs_{session_key}", [])
    
    if not session_recs:
        # Fall back to popular items
        session_recs = cache.get("popular_products", [])
        if not session_recs:
            session_recs = Product.objects.order_by("-rating_number")[:k]
            cache.set("popular_products", [p.parent_asin for p in session_recs], timeout=3600)
    
    return session_recs