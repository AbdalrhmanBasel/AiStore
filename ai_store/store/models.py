from django.db import models
from categories.models import Category
from django.utils.text import slugify
from django.urls import reverse

class Product(models.Model):
    product_name = models.CharField(max_length=200, unique=True, verbose_name='Product Name', help_text='Enter product name (max 200 characters)')
    slug = models.SlugField(max_length=250, unique=True, editable=True, help_text='Auto-generated from product name')
    description = models.TextField(max_length=500, blank=True, verbose_name='Description')
    
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name='Price ($)', default=0.00)
    stock = models.PositiveIntegerField(verbose_name='Stock Quantity', default=0)
    is_available = models.BooleanField(default=True, verbose_name='Available', help_text='Uncheck to hide this product')
    
    images = models.ImageField(upload_to='photos/products/%Y/%m/%d/', verbose_name='Product Image', blank=True, null=True)
    
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products', verbose_name='Category')
    
    created_date = models.DateTimeField(auto_now_add=True)
    modified_date = models.DateTimeField(auto_now=True)
    
    popularity_score = models.FloatField(default=0.0, verbose_name='Popularity Score')
    num_purchases = models.PositiveIntegerField(default=0, verbose_name='Number of Purchases')
    num_views = models.PositiveIntegerField(default=0, verbose_name='Number of Views')
    num_wishlist = models.PositiveIntegerField(default=0, verbose_name='Number of Wishlist Adds')

    class Meta:
        ordering = ['-created_date']
        verbose_name = 'Product'
        verbose_name_plural = 'Products'

    def __str__(self):
        return self.product_name

    def get_absolute_url(self):
        return reverse('product_detail', kwargs={'category_slug': self.category.slug, 'product_slug': self.slug})

    def clean(self):
        if not self.slug:
            base_slug = slugify(self.product_name)
            slug = base_slug
            count = 0
            while Product.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                count += 1
                slug = f"{base_slug}-{count}"
            self.slug = slug

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)
