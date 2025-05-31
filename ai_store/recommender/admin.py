# recommender/admin.py

from django.contrib import admin
from .models import Recommendation

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'score', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__email', 'product__product_name')
