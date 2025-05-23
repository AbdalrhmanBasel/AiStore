from django.contrib import admin
from .models import Category
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils.text import slugify
from django.utils.translation import gettext_lazy as _
from django import forms

class CategoryAdminForm(forms.ModelForm):
    class Meta:
        model = Category
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['slug'].widget.attrs['readonly'] = True  # Makes it look readonly

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    # Auto-populate slug from category name
    form = CategoryAdminForm
    prepopulated_fields = {'slug': ('category_name',)}
    
    # Fields to display in list view
    list_display = (
        'category_name', 
        'slug', 
        'created_at', 
        'updated_at', 
        'image_preview'
    )
    
    # Make category name clickable
    list_display_links = ('category_name',)
    
    # Filter options
    list_filter = ('created_at',)
    
    # Search functionality
    search_fields = ('category_name', 'description')
    
    # Organize fields in admin form
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('category_name', 'description', 'slug')  # ← Added 'slug' here
        }),
        (_('Media'), {
            'fields': ('category_image', 'image_preview')
        }),
        (_('System Info'), {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    # Read-only timestamp and slug fields
    readonly_fields = ('created_at', 'updated_at', 'image_preview')

    def image_preview(self, obj):
        if obj.category_image:
            return format_html(
                '<a href="{}" target="_blank"><img src="{}" width="80" height="80" style="object-fit: cover; border-radius: 4px;" /></a>',
                obj.category_image.url,
                obj.category_image.url
            )
        return mark_safe('<span class="text-muted">No image</span>')
    image_preview.short_description = _('Image Preview')

    # Bulk slug regeneration action
    actions = ['regenerate_slugs']

    def regenerate_slugs(self, request, queryset):
        for obj in queryset:
            obj.slug = slugify(obj.category_name)
            obj.save(update_fields=['slug'])
    regenerate_slugs.short_description = _('Regenerate selected slugs')

