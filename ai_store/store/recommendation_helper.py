# store/recommendation_helper.py

from recommender.client import get_recommendations
from recommender.utils import load_encoders
from store.models import Product

# Load encoders once
user_encoder, product_encoder = load_encoders()

def get_user_encoded(user):
    """Map user.id → encoded"""
    return user_encoder.transform([user.id])[0]

def get_product_from_encoded(encoded_ids):
    """Map encoded product ids → Product queryset"""
    product_ids = product_encoder.inverse_transform(encoded_ids)
    return Product.objects.filter(id__in=product_ids)
