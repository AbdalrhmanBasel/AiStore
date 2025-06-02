# tracking/hooks.py

from tracking.models import InteractionEvent
from django.utils.timezone import now

def log_interaction(user_id, product_id, interaction_type):
    """
    Logs an interaction to the InteractionEvent model.
    """
    # Create interaction event
    InteractionEvent.objects.create(
        user_id=user_id,
        product_id=product_id,
        interaction_type=interaction_type,
        
        timestamp=now()
    )

    print(f"✅ Tracked: user={user_id}, product={product_id}, type={interaction_type}")
