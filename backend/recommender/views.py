# recommender/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from django.core.cache import cache
from .utils.recommend import get_recommendations

class RecommendationsView(APIView):
    def post(self, request):
        user_id = request.data.get("user_id")
        k = int(request.data.get("k", 5))

        if not user_id:
            return Response(
                {"detail": "user_id is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Try getting from cache first
            cached = cache.get(f"recommendations_{user_id}")
            if cached:
                return Response(cached)

            # Get recommendations
            recommended = get_recommendations(user_id=user_id, k=k)
            valid_recommended = [a for a in recommended if a]

            # Cache results for 1 hour
            cache.set(f"recommendations_{user_id}", {
                "user_id": user_id,
                "recommendations": valid_recommended,
                "timestamp": timezone.now().isoformat()
            }, timeout=3600)

            return Response({
                "user_id": user_id,
                "recommendations": valid_recommended,
                "count": len(valid_recommended)
            })
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)