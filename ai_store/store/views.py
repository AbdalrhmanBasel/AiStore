from django.shortcuts import render, get_object_or_404
from .models import Product
from categories.models import Category

def product_detail(request, slug):
    # Get the product based on the slug and render the product detail page
    product = get_object_or_404(Product, slug=slug)
    return render(request, 'product_detail.html', {'product': product})

def store(request, category_slug=None):
    # Initialize products and category_slug
    products = Product.objects.filter(is_available=True)  # All available products by default
    products_count = products.count()  # Get the count of available products

    # If category_slug is provided, filter products by category
    if category_slug:
        category = get_object_or_404(Category, slug=category_slug)
        products = products.filter(category=category)  # Filter products by the selected category
        products_count = products.count()  # Recalculate the product count for the filtered category

    # Pass the products and the total count of available products to the template
    context = {
        'products': products,
        'products_count': products_count,
        'category_slug': category_slug  # Optional: To highlight the active category in the template
    }

    return render(request, 'store/store.html', context)


def product_detail(request, category_slug, product_slug):
    product = get_object_or_404(Product, slug=product_slug, category__slug=category_slug)
    
    context = {
        'product': product,
    }
    return render(request, 'store/product_detail.html', context)
