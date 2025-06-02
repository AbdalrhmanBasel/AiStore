# recommender/management/commands/update_recommendations.py

from django.core.management.base import BaseCommand
from recommender.pipeline import run_full_pipeline

class Command(BaseCommand):
    help = 'Run full GNN pipeline and update recommendations'

    def handle(self, *args, **kwargs):
        run_full_pipeline()
        self.stdout.write(self.style.SUCCESS('✅ Recommendations updated successfully! 🚀'))
