from django.db import models

class Category(models.Model):
    category_name = models.CharField(max_length=255)
    preview_text = models.TextField(max_length=200, verbose_name="Preview Text")
    details_text = models.TextField(max_length=1000, verbose_name="Description")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.category_name

    class Meta:
        verbose_name_plural = 'Categories'
        ordering = ('-created_at',)  

# class Product(models.Model):
#     main_image = models.ImageField(upload_to='Products')
#     product_name = models.CharField(max_length=255)
#     preview_text = models.TextField(max_length=200, verbose_name="Preview Text")
#     details_text = models.TextField(max_length=1000, verbose_name="Description")
#     price = models.DecimalField(max_digits=10, decimal_places=2)
#     old_price = models.DecimalField(max_digits=10, decimal_places=2, blank=True, default=0.00)
#     created_at = models.DateTimeField(auto_now_add=True)
#     category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='category')
#     seller = models.CharField(max_length=255) # TODO: Make Get Seller User Who Created Product

#     def __str__(self):
#         return self.product_name
    
#     class Meta:
#         ordering = ('-created_at',)


class Product(models.Model):
    parent_asin = models.CharField(max_length=20, unique=True)
    title = models.TextField()
    price = models.FloatField()
    average_rating = models.FloatField()
    main_category = models.ForeignKey("Category", on_delete=models.SET_NULL, null=True, blank=True)
    brand = models.CharField(max_length=100, null=True, blank=True)
    first_image = models.URLField(null=True, blank=True)
    
    def __str__(self):
        return self.title