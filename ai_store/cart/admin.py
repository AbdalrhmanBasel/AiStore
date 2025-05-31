from django.contrib import admin
from .models import Cart, CartItem

class CartItemInline(admin.TabularInline):
    model = CartItem
    extra = 0
    readonly_fields = ('subtotal',)
    can_delete = True

class CartAdmin(admin.ModelAdmin):
    list_display = ('cart_id', 'date_added', 'total_items', 'total_price')
    search_fields = ('cart_id',)
    inlines = [CartItemInline]

admin.site.register(Cart, CartAdmin)

@admin.register(CartItem)
class CartItemAdmin(admin.ModelAdmin):
    list_display = ('product', 'cart', 'quantity', 'is_active', 'date_added', 'subtotal')
    list_filter = ('is_active',)
    search_fields = ('product__name', 'cart__cart_id')
    readonly_fields = ('subtotal',)
