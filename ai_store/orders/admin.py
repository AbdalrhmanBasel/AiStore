from django.contrib import admin
from .models import Order, OrderItem, Payment

class OrderItemInline(admin.TabularInline):
    model = OrderItem
    readonly_fields = ('product', 'quantity', 'price')
    extra = 0

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'first_name', 'last_name', 'email', 'total_price', 'status', 'created_at')
    list_filter = ('status', 'created_at', 'country')
    search_fields = ('first_name', 'last_name', 'email', 'phone_number', 'city', 'country')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
    inlines = [OrderItemInline]

@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ('order', 'product', 'quantity', 'price')
    search_fields = ('order__id', 'product__product_name')

@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ('order', 'method', 'status', 'amount_paid', 'transaction_id', 'created_at')
    list_filter = ('method', 'status', 'created_at')
    search_fields = ('order__id', 'transaction_id')
    ordering = ('-created_at',)
