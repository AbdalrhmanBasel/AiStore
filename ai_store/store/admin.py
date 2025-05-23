from django.contrib import admin
from .models import Product
from django.utils.html import format_html
from django.utils.safestring import mark_safe

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    # Auto-populate slug from product name
    prepopulated_fields = {'slug': ('product_name',)}
    
    # Fields to display in list view
    list_display = (
        'product_name', 
        'price_in_dollars', 
        'stock', 
        'category', 
        'modified_date', 
        'is_available',
        'image_preview'
    )
    
    # Make product name clickable
    list_display_links = ('product_name',)
    
    # Filter options
    list_filter = ('is_available', 'category')
    
    # Search functionality
    search_fields = ('product_name', 'description', 'category__category_name')
    
    # Organize fields in admin form
    fieldsets = (
        ('Product Information', {
            'fields': ('product_name', 'description', 'price', 'stock', 'is_available')
        }),
        ('Categorization', {
            'fields': ('category',)
        }),
        ('Media', {
            'fields': ('images',)
        }),
        ('System Info', {
            'fields': ('slug', 'created_date', 'modified_date'),
            'classes': ('collapse',)
        }),
    )
    
    # Read-only timestamp fields
    readonly_fields = ('created_date', 'modified_date')

    # Display price in dollars
    def price_in_dollars(self, obj):
        return f"${obj.price:.2f}" if obj.price else "$0.00"
    price_in_dollars.short_description = 'Price ($)'
    price_in_dollars.admin_order_field = 'price'

    # Image preview
    def image_preview(self, obj):
        if obj.images:
            return format_html(
                '<img src="{}" width="80" height="80" style="object-fit: cover; border-radius: 4px;" />', 
                obj.images.url
            )
        return mark_safe('<span class="text-muted">No image</span>')
    image_preview.short_description = 'Image Preview'

    # Bulk actions
    actions = ['mark_available', 'mark_unavailable']

    def mark_available(self, request, queryset):
        queryset.update(is_available=True)
    mark_available.short_description = "Mark selected products as available"

    def mark_unavailable(self, request, queryset):
        queryset.update(is_available=False)
    mark_unavailable.short_description = "Mark selected products as unavailable"