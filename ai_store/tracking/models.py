# tracking/models.py

from django.db import models
from django.conf import settings
from store.models import Product
from orders.models import Order

# tracking/models.py

class InteractionEvent(models.Model):
    INTERACTION_CHOICES = [
        ('view', 'View'),
        ('cart', 'Add to Cart'),
        ('wishlist', 'Add to Wishlist'),
        ('purchase', 'Purchase'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='interaction_events')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='interaction_events')
    interaction_type = models.CharField(max_length=20, choices=INTERACTION_CHOICES)
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='interaction_events', null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Interaction Event'
        verbose_name_plural = 'Interaction Events'
        indexes = [
            models.Index(fields=['user', 'product', 'interaction_type', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.user} → {self.interaction_type} → {self.product} at {self.timestamp}"
