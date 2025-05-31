# recommender/views.py

from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Recommendation

@login_required
def api_recommendations(request):
    recommendations = Recommendation.objects.filter(user=request.user).order_by('-score')[:10]
    data = [
        {
            'product_id': rec.product.id,
            'product_name': rec.product.product_name,
            'score': rec.score
        }
        for rec in recommendations
    ]
    return JsonResponse({'recommendations': data})
