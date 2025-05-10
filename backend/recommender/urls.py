from django.urls import path
from .views import RecommendationsView

urlpatterns = [
    path('api/recommendations/', RecommendationsView.as_view(), name='recommendations'),
]