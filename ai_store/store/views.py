from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from django.db.models import Q
from .models import Product
from categories.models import Category

def product_detail(request, slug):
    # Get the product based on the slug and render the product detail page
    product = get_object_or_404(Product, slug=slug)
    return render(request, 'product_detail.html', {'product': product})

def store(request, category_slug=None):
    products = Product.objects.filter(is_available=True, stock__gt=0)
    
    query = request.GET.get('q', '').strip()  # search query

    if category_slug:
        category = get_object_or_404(Category, slug=category_slug)
        products = products.filter(category=category)
    
    if query:
        products = products.filter(
            Q(product_name__icontains=query) |
            Q(description__icontains=query) |
            Q(category__category_name__icontains=query)
        ).distinct()

    # Pagination
    paginator = Paginator(products, 6)  # Show 12 products per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'products': page_obj.object_list,
        'products_count': paginator.count,
        'category_slug': category_slug,
        'query': query,
        'page_obj': page_obj,
    }

    return render(request, 'store/store.html', context)

def product_detail(request, category_slug, product_slug):
    product = get_object_or_404(Product, slug=product_slug, category__slug=category_slug)
    
    context = {
        'product': product,
    }
    return render(request, 'store/product_detail.html', context)

