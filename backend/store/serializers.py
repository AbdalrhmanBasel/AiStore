from rest_framework import serializers
from .models import Product, Category


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = [
            'id', 'category_name', 'preview_text', 'details_text',
            'created_at',
        ]

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = [
            'id', 'main_image', 'product_name', 'preview_text',
            'details_text', 'price', 'old_price', 'created_at',
            'category', 'seller',
        ]