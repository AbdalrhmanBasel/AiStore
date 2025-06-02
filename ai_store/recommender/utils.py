# recommender/utils.py 🚀 Optimized with CACHE

import os
import pickle
from store.models import Product

# --- CONFIG ---
ENCODERS_PATH = 'recommender/embeddings/encoders.pkl'

# --- In-memory cache
_cached_encoders = None

# --- Load Encoders ---
def load_encoder():
    global _cached_encoders

    # Use cache if available
    if _cached_encoders is not None:
        return _cached_encoders

    # Load from disk
    if os.path.exists(ENCODERS_PATH):
        with open(ENCODERS_PATH, 'rb') as f:
            enc_bundle = pickle.load(f)
        user_encoder = enc_bundle['user_encoder']
        product_encoder = enc_bundle['product_encoder']
        print(f"✅ Encoders loaded: {len(user_encoder.classes_)} users, {len(product_encoder.classes_)} products")

        # Save to cache
        _cached_encoders = (user_encoder, product_encoder)
        return _cached_encoders
    else:
        print("⚠️ Encoders not found!")
        return None, None

# --- Save Encoders ---
def save_encoders(user_encoder_obj, product_encoder_obj):
    bundle = {
        'user_encoder': user_encoder_obj,
        'product_encoder': product_encoder_obj
    }
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"✅ Encoders saved to {ENCODERS_PATH}")

    # Invalidate cache
    global _cached_encoders
    _cached_encoders = (user_encoder_obj, product_encoder_obj)

# --- Get encoded user_id ---
def get_user_encoded(user, user_encoder):
    if user_encoder is None:
        print("❌ User encoder not loaded")
        return None
    try:
        user_id_enc = user_encoder.transform([user.id])[0]
        return int(user_id_enc)
    except Exception as e:
        print(f"❌ Failed to encode user {user.id}: {e}")
        return None

# --- Map encoded product_id → Product objects ---
def get_products_from_encoded(product_ids_enc, product_encoder):
    if product_encoder is None:
        print("❌ Product encoder not loaded")
        return Product.objects.none()

    try:
        product_ids = product_encoder.inverse_transform(product_ids_enc)
        products = Product.objects.filter(id__in=product_ids, is_available=True)

        products_dict = {p.id: p for p in products}
        ordered_products = [products_dict.get(pid) for pid in product_ids if pid in products_dict]

        return ordered_products

    except Exception as e:
        print(f"❌ Failed to decode product IDs: {e}")
        return Product.objects.none()
