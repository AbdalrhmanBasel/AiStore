from django.urls import path, include
from . import views

from rest_framework.routers import DefaultRouter
from .views import ProductDetail, ProductList, CategoryDetail, CategoryList

app_name = "store"

# router = DefaultRouter()
# router.register(r'products', ProductListViewSet)
# router.register(r'categories', CategoryList)

urlpatterns = [
    path("", views.home_page, name='home_page'),
    path("about/", views.about_page, name="about_page"),
    # path('api/', include(router.urls)),

    # Products
    path('api/products/', views.ProductList.as_view(), name='products_list'),
    path('api/product/<pk>/', views.ProductDetail.as_view(), name="product_detail"),

    # Categories
    path('api/categories/', views.CategoryList.as_view(), name='categories_list'),
    path('api/category/<pk>/', views.CategoryDetail.as_view(), name="category_detail"),
]


