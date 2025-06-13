# recommender/management/commands/generate_interactions.py

from django.core.management.base import BaseCommand
from accounts.models import Account
from store.models import Product
import csv
import random

INTERACTION_CSV = 'recommender/data/interaction_graph.csv'

class Command(BaseCommand):
    help = 'Generate random interaction graph for GNN training'

    def handle(self, *args, **kwargs):
        users = Account.objects.all()
        products = Product.objects.filter(is_available=True)

        if users.count() == 0 or products.count() == 0:
            self.stdout.write(self.style.ERROR("❌ No users or products available!"))
            return

        num_interactions = 50  # You can adjust!

        interactions = []

        for _ in range(num_interactions):
            user = random.choice(users)
            product = random.choice(products)

            interactions.append({
                'user_id': user.id,
                'product_id': product.id
            })

        # Write CSV
        with open(INTERACTION_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['user_id', 'product_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(interactions)

        self.stdout.write(self.style.SUCCESS(f"✅ Generated {len(interactions)} interactions → {INTERACTION_CSV}"))
