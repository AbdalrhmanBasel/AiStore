import os
import sys
import random
import torch
from torch.utils.data import Dataset
import pandas as pd

from logger import get_module_logger

logger = get_module_logger("GraphDataset")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/../../")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class GraphDataset(Dataset):
    def __init__(
        self,
        review_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        negative_sample_ratio: int = 1
    ):
        """
        Constructs a graph dataset for link prediction.

        Args:
            review_df: cleaned reviews with columns 
                       ['user_id','parent_asin','rating','timestamp']
            metadata_df: cleaned metadata indexed by 'parent_asin' with at least
                         ['price','features','categories','average_rating','rating_number']
            val_ratio: fraction of *edges* to reserve for validation
            test_ratio: fraction of *edges* to reserve for test
            negative_sample_ratio: how many negative samples per positive edge
        """
        # 1) dataframes
        self.review_df = review_df.reset_index(drop=True)
        self.metadata_df = metadata_df.set_index("parent_asin")

        # 2) node lists & mappings
        self.user_ids = list(self.review_df['user_id'].unique())
        self.prod_asins = list(self.review_df['parent_asin'].unique())
        self.user2node = {u: i for i, u in enumerate(self.user_ids)}
        self.node2user = {i: u for u, i in self.user2node.items()}
        offset = len(self.user_ids)
        self.prod2node = {p: i + offset for i, p in enumerate(self.prod_asins)}
        self.node2prod = {i: p for p, i in self.prod2node.items()}

        # 3) features dicts
        self._build_user_features()
        self._build_product_features()

        # 4) dims
        self.user_dim = len(next(iter(self.user_feat.values())))
        self.prod_dim = len(next(iter(self.prod_feat.values())))
        self.total_dim = self.user_dim + self.prod_dim

        # 5) build edges, labels, node_features
        self.edges = []
        self.labels = []
        self.node_feats = {}
        self._build_graph()

        # 6) split edges
        self._split_edges(val_ratio, test_ratio)

        # 7) negative sampling
        self._generate_negative_samples(negative_sample_ratio)

    def _build_user_features(self):
        now = pd.Timestamp.now()
        self.user_feat = {}
        for u, grp in self.review_df.groupby('user_id'):
            num   = float(len(grp))
            avg   = float(grp['rating'].mean())
            last  = pd.to_datetime(grp['timestamp']).max()
            days  = float((now - last).days)
            self.user_feat[u] = {'num_reviews':num, 'avg_rating':avg, 'days_since_last':days}

    def _build_product_features(self):
        self.prod_feat = {}
        for asin in self.prod_asins:
            if asin in self.metadata_df.index:
                rec    = self.metadata_df.loc[asin]
                price  = float(rec.get('price',0.0))
                feats  = rec.get('features',[])
                cats   = rec.get('categories',[])
                avg    = float(rec.get('average_rating',0.0))
                cnt    = float(rec.get('rating_number',0.0))
                self.prod_feat[asin] = {
                    'price':price,
                    'features_count':float(len(feats)),
                    'categories_count':float(len(cats)),
                    'avg_rating':avg,
                    'rating_number':cnt
                }
            else:
                logger.warning(f"ASIN {asin} not in metadata → using zeros")
                self.prod_feat[asin] = {'price':0.,'features_count':0.,'categories_count':0.,
                                        'avg_rating':0.,'rating_number':0.}

    def _build_graph(self):
        pad_u = torch.zeros(self.user_dim, dtype=torch.float32)
        pad_p = torch.zeros(self.prod_dim, dtype=torch.float32)
        ulog = plog = 0
        MAX_LOG = 5

        for _, row in self.review_df.iterrows():
            u, p, r = row['user_id'], row['parent_asin'], float(row['rating'])
            ui, pi  = self.user2node[u], self.prod2node[p]
            # bidir edge + label
            self.edges += [[ui,pi],[pi,ui]]
            self.labels+= [r,r]

            # user feat
            if ui not in self.node_feats:
                uf = self.user_feat[u]
                ut = torch.tensor([uf['num_reviews'],uf['avg_rating'],uf['days_since_last']],dtype=torch.float32)
                self.node_feats[ui] = torch.cat([ut, pad_u.new_zeros(self.prod_dim)])
                if ulog<MAX_LOG:
                    logger.info(f"User node {u}(id={ui}) feats: {ut.tolist()}")
                    ulog+=1

            # prod feat
            if pi not in self.node_feats:
                pf = self.prod_feat[p]
                pt = torch.tensor([pf['price'],pf['features_count'],pf['categories_count'],
                                   pf['avg_rating'],pf['rating_number']],dtype=torch.float32)
                self.node_feats[pi] = torch.cat([pad_p.new_zeros(self.user_dim), pt])
                if plog<MAX_LOG:
                    logger.info(f"Product node {p}(id={pi}) feats: {pt.tolist()}")
                    plog+=1

    def _split_edges(self, val_ratio, test_ratio):
        n = len(self.edges)
        idx = list(range(n))
        random.shuffle(idx)
        n_val  = int(val_ratio  * n)
        n_test = int(test_ratio * n)
        self.val_idx  = idx[:n_val]
        self.test_idx = idx[n_val:n_val+n_test]
        self.train_idx= idx[n_val+n_test:]
        logger.info(f"Edges → train={len(self.train_idx)}, val={len(self.val_idx)}, test={len(self.test_idx)}")

    def _generate_negative_samples(self, neg_ratio):
        pos = set(map(tuple, self.edges))
        self.neg_edges = []
        for _ in range(neg_ratio * len(self.edges)):
            while True:
                u = random.choice(self.user_ids)
                p = random.choice(self.prod_asins)
                e = (self.user2node[u], self.prod2node[p])
                if e not in pos:
                    self.neg_edges.append(e)
                    break
        logger.info(f"Generated {len(self.neg_edges)} negative edges")

    def __len__(self):
        return len(self.train_idx)

    def __getitem__(self, idx):
        # alternate pos/neg
        if random.random()<0.5:
            src,dst = self.edges[self.train_idx[idx]]
            label    = torch.tensor(self.labels[self.train_idx[idx]],dtype=torch.float32)
        else:
            src,dst = self.neg_edges[idx % len(self.neg_edges)]
            label    = torch.tensor(0.0,dtype=torch.float32)
        return torch.tensor([[src],[dst]],dtype=torch.long), label

    def get_all_edges(self):
        return torch.tensor(self.edges, dtype=torch.long).t().contiguous()

    def get_all_node_features(self):
        N = len(self.user_ids)+len(self.prod_asins)
        return torch.stack([self.node_feats[i] for i in range(N)],dim=0)

    def get_all_labels(self):
        return torch.tensor(self.labels, dtype=torch.float32)

    def get_graph_data(self):
        return self.get_all_edges(), self.get_all_node_features(), self.get_all_labels()

    def get_mappings(self):
        return {
            'user2node': self.user2node,
            'node2user': self.node2user,
            'prod2node': self.prod2node,
            'node2prod': self.node2prod
        }
