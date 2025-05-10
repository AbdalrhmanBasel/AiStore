from django.urls import path
from . import views

urlpatterns = [
    path('api/recommendations/<int:user_id>/', views.recommend, name='recommend'),
]