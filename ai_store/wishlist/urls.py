from django.urls import path
from . import views

urlpatterns = [
    path('', views.wishlist, name='wishlist'),
    path('add/<int:product_id>/', views.add_to_wishlist, name='add_to_wishlist'),
    path('remove/<int:item_id>/', views.remove_from_wishlist, name='remove_from_wishlist'),
    path('move-to-cart/<int:item_id>/', views.move_to_cart, name='move_to_cart'),
    path('move-all-to-cart/', views.move_all_to_cart, name='move_all_to_cart'),


]
