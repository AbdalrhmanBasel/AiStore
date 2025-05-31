from django.contrib import admin
from .models import WishlistItem

@admin.register(WishlistItem)
class WishlistItemAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'product', 'added_at')
    search_fields = ('user__email', 'user__username', 'product__product_name')
    list_filter = ('added_at',)
    ordering = ('-added_at',)
