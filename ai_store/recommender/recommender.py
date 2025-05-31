# recommender/recommender.py

"""
Example GNN pipeline — replace with your actual model
"""

import random
from recommender.models import Recommendation
from store.models import Product
from accounts.models import Account

def run_gnn_and_update():
    print("Running GNN model...")

    # Example: get users and products
    users = Account.objects.filter(is_active=True)
    products = Product.objects.filter(is_available=True)

    for user in users:
        top_products = random.sample(list(products), min(5, len(products)))  # Simulate top 5 recommendations

        for product in top_products:
            score = random.uniform(0.7, 1.0)  # Simulate GNN score

            Recommendation.objects.update_or_create(
                user=user,
                product=product,
                defaults={'score': score}
            )

    print("GNN recommendations updated.")
