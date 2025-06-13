# recommender/data/generate_interactions.py

import csv
import random

from datetime import datetime, timedelta
import django

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()



from store.models import Product
from django.contrib.auth import get_user_model

User = get_user_model()

# CONFIG
OUTPUT_PATH = 'recommender/data/interaction_graph.csv'
NUM_INTERACTIONS = 10000  # How many interactions to generate
INTERACTION_TYPES = ['view', 'wishlist', 'purchase']

def generate_interactions():
    users = list(User.objects.all())
    products = list(Product.objects.filter(is_available=True))

    if not users:
        print("❌ No users found!")
        return

    if not products:
        print("❌ No products found!")
        return

    print(f"✅ Generating {NUM_INTERACTIONS} interactions for {len(users)} users and {len(products)} products...")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'product_id', 'interaction_type', 'timestamp'])

        for _ in range(NUM_INTERACTIONS):
            user = random.choice(users)
            product = random.choice(products)
            interaction = random.choice(INTERACTION_TYPES)
            timestamp = datetime.now() - timedelta(days=random.randint(0, 180))  # Last 6 months

            writer.writerow([
                user.id,
                product.id,
                interaction,
                timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ])

    print(f"🎉 Done! Interactions saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    generate_interactions()
