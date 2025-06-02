# recommender/client.py

import requests
from recommender.utils import get_user_encoded, get_products_from_encoded

RECOMMEND_API_URL = "http://127.0.0.1:8005/recommend/"  # match api_service.py

def get_recommendations(user, top_n=10):
    user_id_enc = get_user_encoded(user)
    if user_id_enc is None:
        print("❌ Cannot recommend: user encoding failed")
        return []

    payload = {
        "user_id_enc": user_id_enc,
        "top_n": top_n
    }

    try:
        response = requests.post(RECOMMEND_API_URL, json=payload)
        response.raise_for_status()

        product_ids_enc = response.json()
        print(f"✅ Got recommendations: {product_ids_enc}")

        products = get_products_from_encoded(product_ids_enc)
        print(f"✅ Resolved {len(products)} Product objects")

        return products

    except Exception as e:
        print(f"❌ Failed to get recommendations: {e}")
        return []
