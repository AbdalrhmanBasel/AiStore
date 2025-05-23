# recommender/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.apps import apps
from django.core.cache import cache
from django.db.models import Q
from store.models import Product
from django.utils import timezone
from django.core.exceptions import ValidationError
import numpy as np

class RecommendationsView(APIView):
    """
    POST /api/recommender/recommend/
    Expected JSON Body:
    {
        "user_id": "AGBFYI2DDIKXC5Y4FARTYDTQBMFQ",
        "k": 5
    }
    Returns:
    {
        "type": "personalized|new_user|anonymous|fallback",
        "recommendations": ["B07BSVLHYD", "B07RBBN3X3", ...],
        "products": [
            {
                "parent_asin": "B07BSVLHYD",
                "title": "Fintie Slimshell Case",
                "price": "14.99",
                "average_rating": 4.6
            },
            ...
        ],
        "timestamp": "2025-05-11T12:34:56.789Z"
    }
    """
    
    def post(self, request):
        """
        Handle recommendation requests for all user types:
        - Authenticated users: use GNN model
        - New users: fallback to popularity
        - Anonymous users: session-based recommendations
        """
        try:
            # Extract request data
            user_id = request.data.get("user_id")
            k = int(request.data.get("k", 5))
            
            if not user_id:
                # 🧸 Anonymous User
                return self._get_anonymous_recommendations(request, k=k)

            # Load recommender system
            app_config = apps.get_app_config('recommender')
            full_data, gnn, predictor = app_config.full_data, app_config.gnn, app_config.predictor
            
            # Validate user exists in mappings
            if str(user_id) not in app_config.item_id_map:
                # 🚫 New User: fallback to popularity
                return self._get_new_user_recommendations(k=k)

            # 🧠 Existing User: Use GNN for personalized recommendations
            user_idx = int(app_config.item_id_map[str(user_id)])
            emb = gnn(full_data.x_dict, full_data.edge_index_dict)

            # Predict scores
            user_emb = emb['user'][user_idx].unsqueeze(0)
            item_embs = emb['item']
            scores = predictor(
                user_emb.expand(item_embs.size(0), -1),
                item_embs
            ).cpu().numpy()

            # Exclude interacted items
            edge_index = full_data['user', 'rates', 'item'].edge_index
            interacted_items = edge_index[1][edge_index[0] == user_idx].cpu().numpy()
            scores[interacted_items] = -np.inf

            # Get top-k item indices
            top_k_indices = np.argpartition(scores, -k)[-k:]
            recommended = [app_config.id_to_item.get(str(int(i)), None) for i in top_k_indices]
            valid_recommended = [a for a in recommended if a]

            # Hydrate with product metadata
            product_data = self._get_product_data(valid_recommended)

            return Response({
                "user_id": user_id,
                "type": "personalized",
                "recommendations": valid_recommended,
                "products": product_data,
                "count": len(valid_recommended),
                "timestamp": timezone.now().isoformat()
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # 🚨 Fallback on error
            fallback_recs = self._get_fallback_recommendations(k=k)
            return Response({
                "user_id": user_id,
                "type": "fallback",
                "error": str(e),
                "recommendations": fallback_recs,
                "products": self._get_product_data(fallback_recs),
                "timestamp": timezone.now().isoformat()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _get_anonymous_recommendations(self, request, k=5):
        """Return session-based recommendations for anonymous users"""
        session_key = request.session.session_key or request.session.create()
        session_recs = cache.get(f"session_recs_{session_key}", [])
        
        if not session_recs:
            # ⏳ Fall back to popular products
            session_recs = cache.get("popular_products", [])
            if not session_recs:
                # 🔄 Final fallback: random from metadata
                session_recs = Product.objects.values_list('parent_asin', flat=True).order_by("-average_rating")[:k]
                cache.set(f"session_recs_{session_key}", list(session_recs), 3600)
        
        product_data = self._get_product_data(session_recs)
        return Response({
            "type": "anonymous",
            "recommendations": session_recs,
            "products": product_data,
            "session_key": session_key,
            "timestamp": timezone.now().isoformat()
        }, status=status.HTTP_200_OK)

    def _get_new_user_recommendations(self, k=5):
        """Fallback for users not in mapping"""
        popular_recs = cache.get("popular_products", [])
        if not popular_recs:
            # 📦 Fallback: top products by rating
            popular_recs = Product.objects.values_list('parent_asin', flat=True).order_by("-average_rating")[:k]
            cache.set("popular_products", list(popular_recs), 3600)
        
        product_data = self._get_product_data(popular_recs)
        return Response({
            "type": "new_user",
            "recommendations": popular_recs,
            "products": product_data,
            "timestamp": timezone.now().isoformat()
        }, status=status.HTTP_200_OK)

    def _get_fallback_recommendations(self, k=5):
        """Emergency fallback when GNN fails"""
        try:
            # Try random from valid ASINs
            return Product.objects.values_list('parent_asin', flat=True).order_by("?")[:k]
        except Exception:
            return ["B07BSVLHYD", "B07RBBN3X3", "B07R217P25"]

    def _get_product_data(self, asins):
        """Hydrate ASINs with product metadata"""
        if not asins:
            return []
        
        try:
            products = Product.objects.filter(parent_asin__in=asins)
            return [{
                "parent_asin": p.parent_asin,
                "title": p.title,
                "price": str(p.price),
                "average_rating": p.average_rating,
                "main_category": p.main_category.name if p.main_category else "Unknown",
                "brand": p.brand,
                "first_image": p.first_image
            } for p in products]
        except Exception as e:
            print(f"⚠️ Product hydration failed: {e}")
            return []