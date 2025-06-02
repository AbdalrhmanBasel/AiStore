# recommender/management/commands/export_graph.py

from django.core.management.base import BaseCommand
from recommender.graph.graph_export import export_interaction_graph


class Command(BaseCommand):
    help = 'Export interaction graph CSV'

    def handle(self, *args, **kwargs):
        export_interaction_graph()
        self.stdout.write(self.style.SUCCESS('Graph exported successfully'))
