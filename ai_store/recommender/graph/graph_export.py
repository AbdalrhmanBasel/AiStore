# recommender/graph/graph_export.py

import csv
from tracking.models import InteractionEvent

OUTPUT_FILE = 'recommender/data/interaction_graph.csv'

def export_interaction_graph():
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'product_id', 'interaction_type', 'timestamp'])

        # Export all InteractionEvents
        events = InteractionEvent.objects.all().order_by('timestamp')

        for event in events:
            writer.writerow([
                event.user.id,
                event.product.id,
                event.interaction_type,
                event.timestamp
            ])

    print(f"✅ Graph exported to {OUTPUT_FILE}")
