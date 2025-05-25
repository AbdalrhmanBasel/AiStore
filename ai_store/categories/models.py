from django.db import models
from django.utils.text import slugify
from django.urls import reverse
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

class Category(models.Model):
    # Basic Fields
    category_name = models.CharField(
        max_length=50,
        verbose_name=_('Category Name'),
        help_text=_('Enter the category name (max 50 characters)')
    )
    
    slug = models.SlugField(
        max_length=100,
        unique=True,
        blank=True,
        editable=True,  # Allow slug to be editable in admin
        verbose_name=_('URL Slug'),
        help_text=_('Auto-generated from category name')
    )
    
    description = models.TextField(
        max_length=255,
        blank=True,
        verbose_name=_('Description'),
        help_text=_('Short description (max 255 characters)')
    )
    
    # Media Fields
    category_image = models.ImageField(
        upload_to='categories/%Y/%m/%d/',  # Organized by date
        blank=True,
        verbose_name=_('Category Image'),
        help_text=_('Upload a category image (optional)')
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('Category')
        verbose_name_plural = _('Categories')
        ordering = ['category_name']
        indexes = [
            models.Index(fields=['slug']),
        ]

    def __str__(self):
        return self.category_name

    def get_absolute_url(self):
        """Return URL for the category's products list page."""
        return reverse('products_by_category', kwargs={'category_slug': self.slug})

    def get_url(self):
        return self.get_absolute_url()

    def clean(self):
        """Ensure slug uniqueness and auto-generation"""
        if not self.slug:
            self.slug = slugify(self.category_name)
        
        if self.slug and Category.objects.filter(slug=self.slug).exclude(pk=self.pk).exists():
            raise ValidationError({'slug': _('A category with this slug already exists.')})

    def save(self, *args, **kwargs):
        """Auto-generate slug and validate before saving"""
        if not self.slug:
            base_slug = slugify(self.category_name)
            slug = base_slug
            count = 0
            while Category.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                count += 1
                slug = f"{base_slug}-{count}"
            self.slug = slug
        
        self.full_clean()  # run validations including unique slug check
        super().save(*args, **kwargs)
