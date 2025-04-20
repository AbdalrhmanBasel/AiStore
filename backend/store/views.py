from django.shortcuts import render
from .serializers import ProductSerializer, CategorySerializer
from .models import Product, Category
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView

# Create your views here.

def home_page(request):
    return render (request, "store/home_page.html")

def about_page(request):
    return render (request, "store/about_page.html")


# DRF
class ProductList(ListView):
    """
    TODO: The goal is to list products
    1. Instead of using ViewSet, use appropiriate
    type of class based views for listing.
    """
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class ProductDetail(LoginRequiredMixin, DetailView):
    """
    TODO: The Goal is to send Product Details APIs
    To frontend, ensure that is happening successfully.
    Why LoginRequiredMixin is here?
    """
    model = Product
    

class CategoryList(ListView):
    """
    TODO: The goal is to list of categories
    1. Instead of using ViewSet, use appropiriate
    type of class based views for listing.
    """
    queryset = Category.objects.all()
    serializer_class = CategorySerializer

class CategoryDetail(LoginRequiredMixin, DetailView):
    """
    TODO: The Goal is to send Product Details APIs
    To frontend, ensure that is happening successfully.
    Why LoginRequiredMixin is here?
    """
    model = Category