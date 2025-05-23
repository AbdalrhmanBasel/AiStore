# recommender/urls.py
from django.urls import path
from .views import RecommendationsView

urlpatterns = [
    path('recommend/', RecommendationsView.as_view(), name='recommendations'),
]

