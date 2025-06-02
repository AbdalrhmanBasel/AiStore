# tracking/admin.py

from django.contrib import admin
from .models import InteractionEvent

@admin.register(InteractionEvent)
class InteractionEventAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'interaction_type', 'order', 'timestamp')
    list_filter = ('interaction_type', 'timestamp')
    search_fields = ('user__email', 'product__product_name', 'interaction_type')
