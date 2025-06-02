# ratings/models.py

from django.db import models
from django.conf import settings
from store.models import Product

class Rating(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='ratings')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='ratings')
    rating = models.PositiveSmallIntegerField(
        verbose_name='Rating (1-5)',
        choices=[(i, str(i)) for i in range(1, 6)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Rating'
        verbose_name_plural = 'Ratings'
        unique_together = ('user', 'product')  # One rating per user-product

    def __str__(self):
        return f"{self.user} → {self.product} ({self.rating} ⭐️)"




class Review(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='reviews')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    comment = models.TextField(verbose_name='Comment')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Review'
        verbose_name_plural = 'Reviews'

    def __str__(self):
        return f"{self.user} → {self.product} (Review)"
