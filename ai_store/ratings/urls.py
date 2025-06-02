# ratings/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('rating/add/<int:product_id>/', views.add_rating, name='add_rating'),
    path('review/add/<int:product_id>/', views.add_review, name='add_review'),
]
