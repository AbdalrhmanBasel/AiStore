# recommender/management/commands/import_categories.py

import csv
import os
from django.core.management.base import BaseCommand
from categories.models import Category
from django.utils.text import slugify

class Command(BaseCommand):
    help = 'Import categories from CSV'

    def handle(self, *args, **kwargs):
        csv_path = 'recommender/data/categories.csv'

        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"CSV file not found: {csv_path}"))
            return

        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                category, created = Category.objects.get_or_create(
                    category_name=row['category_name'],
                    slug=slugify(row['slug'])
                )
                if created:
                    self.stdout.write(f"✅ Created category: {category.category_name}")
                else:
                    self.stdout.write(f"ℹ️ Category already exists: {category.category_name}")

        self.stdout.write(self.style.SUCCESS("🎉 Categories import complete!"))
