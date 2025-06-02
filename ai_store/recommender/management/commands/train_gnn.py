# recommender/management/commands/train_gnn.py

from django.core.management.base import BaseCommand
from recommender.gnn.train import train_gnn

class Command(BaseCommand):
    help = 'Train GNN and save embeddings'

    def handle(self, *args, **kwargs):
        train_gnn()
        self.stdout.write(self.style.SUCCESS('✅ GNN trained and embeddings saved'))
