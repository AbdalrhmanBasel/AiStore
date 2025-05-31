# recommender/management/commands/update_recommendations.py

from django.core.management.base import BaseCommand
from recommender.recommender import run_gnn_and_update

class Command(BaseCommand):
    help = 'Run GNN model and update recommendations'

    def handle(self, *args, **kwargs):
        run_gnn_and_update()
        self.stdout.write(self.style.SUCCESS('Recommendations updated successfully'))



# python manage.py update_recommendations
