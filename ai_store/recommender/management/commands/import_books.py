import csv
import os
from decimal import Decimal, ROUND_HALF_UP
from django.core.management.base import BaseCommand
from categories.models import Category
from store.models import Product

CSV_FILE_PATH = 'recommender/data/books.csv'

class Command(BaseCommand):
    help = 'Import books from CSV into Product model'

    def handle(self, *args, **kwargs):
        if not os.path.exists(CSV_FILE_PATH):
            self.stdout.write(self.style.ERROR(f"CSV file not found: {CSV_FILE_PATH}"))
            return

        with open(CSV_FILE_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                try:
                    category = Category.objects.get(slug=row['category_slug'])

                    product, created = Product.objects.update_or_create(
                        slug=row['slug'],
                        defaults={
                            'product_name': row['product_name'],
                            'description': row['description'],
                            'price': Decimal(row['price']).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
                            'stock': int(row['stock']),
                            'is_available': row['is_available'].strip().upper() == 'TRUE',
                            'category': category,
                            'popularity_score': float(row['popularity_score']),
                            'num_purchases': int(row['num_purchases']),
                            'num_views': int(row['num_views']),
                            'num_wishlist': int(row['num_wishlist']),
                        }
                    )

                    if created:
                        self.stdout.write(self.style.SUCCESS(f"✅ Created product: {product.product_name}"))
                    else:
                        self.stdout.write(self.style.WARNING(f"Updated product: {product.product_name}"))

                except Category.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f"❌ Category not found: {row['category_slug']}"))

        self.stdout.write(self.style.SUCCESS("🎉 Books import complete!"))
