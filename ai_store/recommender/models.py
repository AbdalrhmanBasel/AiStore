# recommender/models.py

from django.db import models
from django.conf import settings
from store.models import Product

class Recommendation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'product')
        ordering = ['-score', '-created_at']

    def __str__(self):
        return f"Recommendation: {self.user.email} -> {self.product.product_name} (score: {self.score:.2f})"
