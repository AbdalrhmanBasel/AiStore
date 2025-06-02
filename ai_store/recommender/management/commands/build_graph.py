# recommender/management/commands/build_graph.py

from django.core.management.base import BaseCommand
from recommender.graph.loader import build_graph_data

class Command(BaseCommand):
    help = 'Build PyG graph from interaction CSV'

    def handle(self, *args, **kwargs):
        build_graph_data()
        self.stdout.write(self.style.SUCCESS('Graph data built successfully'))
