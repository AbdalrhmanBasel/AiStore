from django.db import models
from store.models import Product
from django.conf import settings

class Cart(models.Model):
    cart_id = models.CharField(
        max_length=50,
        unique=True,
        blank=True,
        editable=False,
        help_text="Unique identifier for the cart"
    )
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    date_added = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    def __str__(self):
        return self.cart_id or f"Cart {self.pk}"

    def save(self, *args, **kwargs):
        if not self.cart_id:
            import uuid
            self.cart_id = str(uuid.uuid4())
        super().save(*args, **kwargs)

    def total_price(self):
        return sum(item.subtotal() for item in self.items.filter(is_active=True))

    def total_items(self):
        return sum(item.quantity for item in self.items.filter(is_active=True))



class CartItem(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='cart_items')
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    quantity = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    date_added = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    def __str__(self):
        return f"{self.quantity} x {self.product.product_name}"

    def subtotal(self):
        return self.product.price * self.quantity

    class Meta:
        unique_together = ('product', 'cart')
        ordering = ['-date_added']


