import argparse
import pickle
import time
from typing import List, Tuple, Dict, Any

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.additional_datasets import FBForum, IAContact, IAContactsHypertext2009, IAEmailEU, IARadoslawEmail, \
    SocSignBitcoinAlpha, WikiElections, AlibabaData
from stellargraph import StellarGraph
from temporal_walk import TemporalWalk
from stellargraph.data import TemporalRandomWalk, BiasedRandomWalk
from stellargraph.datasets import IAEnronEmployees

# Constants as per paper
EMBEDDING_SIZE = 128
NUM_WALKS_PER_NODE = 10
WALK_LENGTH = 80
DEFAULT_CONTEXT_WINDOW_SIZE = 10
TRAIN_RATIO = 0.75
WORKERS=4


class TemporalLinkPredictor:
    def __init__(self, embedding_params: Dict[str, Any] = None):
        self.embedding_params = embedding_params or {
            'embedding_size': EMBEDDING_SIZE,
            'num_walks': NUM_WALKS_PER_NODE,
            'walk_length': WALK_LENGTH,
            'context_window': DEFAULT_CONTEXT_WINDOW_SIZE,
            'walk_bias': 'Exponential',
            'initial_edge_bias': 'Uniform',
            'edge_operator': 'all',
            'weight_node2vec': False,
            'is_directed': False
        }

    @staticmethod
    def split_edges_temporal(edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Temporal split using first 75% edges for training"""
        edges_sorted = edges.sort_values('time')
        n_train = int(len(edges_sorted) * TRAIN_RATIO)
        return edges_sorted[:n_train], edges_sorted[n_train:]

    def generate_new_temporal_walks(self, edges_list: List) -> List[List[str]]:
        """Generate walks using new temporal walk method"""
        temporal_walk = TemporalWalk(is_directed=self.embedding_params['is_directed'])

        temporal_walk.add_multiple_edges(edges_list)
        walks = temporal_walk.get_random_walks_for_all_nodes(
            max_walk_len=self.embedding_params['walk_length'],
            walk_bias=self.embedding_params['walk_bias'],
            num_walks_per_node=self.embedding_params['num_walks'],
            initial_edge_bias=self.embedding_params['initial_edge_bias'],
            walk_direction="Forward_In_Time"
        )
        return [[str(node) for node in walk] for walk in walks]

    def generate_old_temporal_walks(self, graph: StellarGraph, num_cw: int) -> List[List[str]]:
        """Generate walks using old temporal walk method"""
        temporal_rw = TemporalRandomWalk(graph)
        return temporal_rw.run(
            num_cw=num_cw,
            cw_size=self.embedding_params['context_window'],
            max_walk_length=self.embedding_params['walk_length'],
            walk_bias=self.embedding_params['walk_bias'].lower()
        )

    def train_node2vec_model(self, graph: nx.Graph) -> (Word2Vec, float):
        """Train Node2Vec and return the skip-gram model"""
        start_time = time.time()
        node2vec = Node2Vec(
            graph,
            dimensions=self.embedding_params['embedding_size'],
            walk_length=self.embedding_params['walk_length'],
            num_walks=self.embedding_params['num_walks'],
            weight_key='time',
            workers=WORKERS
        )
        walk_time = time.time() - start_time
        return (
            node2vec.fit(
                window=self.embedding_params['context_window'],
                min_count=0,
                epochs=10
            ),
            walk_time
        )

    def learn_embeddings(self, walks: List[List[str]]) -> Word2Vec:
        """Learn embeddings using Word2Vec"""
        return Word2Vec(
            walks,
            vector_size=self.embedding_params['embedding_size'],
            window=self.embedding_params['context_window'],
            min_count=0,
            sg=1,
            workers=WORKERS,
            epochs=10
        )

    @staticmethod
    def prepare_link_prediction_data(graph: StellarGraph, test_edges: pd.DataFrame) -> Tuple[List[Tuple], np.ndarray]:
        """Prepare positive and negative examples"""
        positive_edges = list(test_edges[["source", "target"]].itertuples(index=False))

        def sample_negative_edges(n_samples: int) -> List[Tuple]:
            nodes = list(graph.nodes())
            positive_set = set(positive_edges)
            negative_edges = []
            while len(negative_edges) < n_samples:
                src, tgt = np.random.choice(nodes, 2, replace=False)
                if (src, tgt) not in positive_set and (tgt, src) not in positive_set:
                    negative_edges.append((src, tgt))
            return negative_edges

        negative_edges = sample_negative_edges(len(positive_edges))
        edges = positive_edges + negative_edges
        labels = np.array([1] * len(positive_edges) + [0] * len(negative_edges))

        return edges, labels

    def compute_edge_features(self, edges: List[Tuple], model: Word2Vec, operator: str) -> np.ndarray:
        """Compute edge features using Hadamard product"""
        features = []
        for src, tgt in edges:
            try:
                src_emb = model.wv[str(src)]
                tgt_emb = model.wv[str(tgt)]

                if operator == 'weighted-L1':
                    edge_emb = np.abs(src_emb - tgt_emb)
                elif operator == 'weighted-L2':
                    edge_emb = (src_emb - tgt_emb) ** 2
                elif operator == 'average':
                    edge_emb = (src_emb + tgt_emb) / 2.0
                elif operator == 'hadamard':
                    edge_emb = src_emb * tgt_emb
                else:
                    raise ValueError(f"Unknown operator: {operator}")

                features.append(edge_emb)
            except KeyError:
                features.append(np.zeros(model.vector_size))
        return np.array(features)

    def evaluate_embeddings(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate embeddings using logistic regression"""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        clf = LogisticRegressionCV(
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            max_iter=1000,
            solver='liblinear'
        )
        clf.fit(features_scaled, labels)

        predictions = clf.predict_proba(features_scaled)
        return roc_auc_score(labels, predictions[:, 1])

    def run_evaluation(self, graph: StellarGraph, edges: pd.DataFrame, run_number) -> Dict[str, Any]:
        """Run complete evaluation for all three methods"""
        # Split edges
        train_edges, test_edges = self.split_edges_temporal(edges)

        # Create training graph
        train_graph = StellarGraph(
            nodes=pd.DataFrame(index=graph.nodes()),
            edges=train_edges,
            edge_weight_column="time",
            is_directed=self.embedding_params['is_directed']
        )

        # Prepare test data
        test_edges_list, labels = self.prepare_link_prediction_data(train_graph, test_edges)

        # Number of walks
        num_cw = len(graph.nodes()) * self.embedding_params['num_walks'] * (self.embedding_params['walk_length'] - self.embedding_params['context_window'] + 1)

        # Convert edges for new temporal walk
        edges_list = [(int(row[0]), int(row[1]), row[2]) for row in train_edges.to_numpy()]

        # Generate networkx graph for node2vec
        graph = nx.Graph() if not self.embedding_params['is_directed'] else nx.DiGraph()
        for _, row in train_edges.iterrows():
            if self.embedding_params['weighted_node2vec']:
                graph.add_edge(row["source"], row["target"], time=row["time"])
            else:
                graph.add_edge(row["source"], row["target"])

        results = {}


        operators = ['weighted-L1', 'weighted-L2', 'average', 'hadamard']

        if self.embedding_params['edge_operator'] == 'all':
            selected_operator = operators[run_number % len(operators)]
        else:
            selected_operator = self.embedding_params['edge_operator']

        # 1. New Temporal Walk
        start_time = time.time()
        new_temporal_walks = self.generate_new_temporal_walks(edges_list)
        new_temporal_walk_sampling_time = time.time() - start_time
        new_model = self.learn_embeddings(new_temporal_walks)
        new_features = self.compute_edge_features(test_edges_list, new_model, selected_operator)
        new_auc = self.evaluate_embeddings(new_features, labels)
        results['new_temporal'] = {
            'auc': new_auc,
            'time': new_temporal_walk_sampling_time
        }

        # 2. Old Temporal Walk
        start_time = time.time()
        old_temporal_walks = self.generate_old_temporal_walks(train_graph, num_cw)
        old_temporal_walk_sampling_time = time.time() - start_time
        old_model = self.learn_embeddings(old_temporal_walks)
        old_features = self.compute_edge_features(test_edges_list, old_model, selected_operator)
        old_auc = self.evaluate_embeddings(old_features, labels)
        results['old_temporal'] = {
            'auc': old_auc,
            'time': old_temporal_walk_sampling_time
        }

        # 3. Node2Vec
        node2vec_model, node2vec_walk_time = self.train_node2vec_model(graph)
        node2vec_features = self.compute_edge_features(test_edges_list, node2vec_model, selected_operator)
        node2vec_auc = self.evaluate_embeddings(node2vec_features, labels)
        results['node2vec'] = {
            'auc': node2vec_auc,
            'time': node2vec_walk_time
        }

        return results


def get_dataset(dataset_name):
    if dataset_name == 'fb_forum':
        return FBForum()
    elif dataset_name == 'ia_contact':
        return IAContact()
    elif dataset_name == 'ia_contacts_hypertext_2009':
        return IAContactsHypertext2009()
    elif dataset_name == 'ia_email_eu':
        return IAEmailEU()
    elif dataset_name == 'ia_enron_employees':
        return IAEnronEmployees()
    elif dataset_name == 'ia_radoslaw_email':
        return IARadoslawEmail()
    elif dataset_name == 'soc_sign_bitcoin_alpha':
        return SocSignBitcoinAlpha()
    elif dataset_name == 'wiki_elections':
        return WikiElections()
    elif dataset_name == 'alibaba':
        return AlibabaData()
    else:
        raise ValueError(f'Invalid dataset name {dataset_name}')


def main():
    parser = argparse.ArgumentParser(description="Temporal Link Prediction Comparison")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--walk_bias', type=str, default="Exponential")
    parser.add_argument('--initial_edge_bias', type=str, default="Uniform")
    parser.add_argument('--p', type=float, default=1.0, help='Return parameter for node2vec')
    parser.add_argument('--q', type=float, default=1.0, help='In-out parameter for node2vec')
    parser.add_argument('--n_runs', type=int, default=12)
    parser.add_argument('--context_window_size', type=int, default=-1)
    parser.add_argument('--edge_operator', type=str, default='all')
    parser.add_argument('--weighted_node2vec', action='store_true', help='Whether to run weighted node2vec')
    parser.add_argument('--directed', action='store_true', help='Is a directed dataset')

    args = parser.parse_args()


    dataset = get_dataset(args.dataset)
    graph, edges = dataset.load()

    context_window = DEFAULT_CONTEXT_WINDOW_SIZE if args.context_window_size == -1 else args.context_window_size

    # Setup parameters
    params = {
        'embedding_size': EMBEDDING_SIZE,
        'num_walks': NUM_WALKS_PER_NODE,
        'walk_length': WALK_LENGTH,
        'context_window': context_window,
        'walk_bias': args.walk_bias,
        'initial_edge_bias': args.initial_edge_bias,
        'p': args.p,
        'q': args.q,
        'edge_operator': args.edge_operator,
        'weighted_node2vec': args.weighted_node2vec,
        'is_directed': args.directed
    }

    predictor = TemporalLinkPredictor(params)

    # Run multiple trials
    all_results = []
    for run in range(args.n_runs):
        print(f"\nRun {run + 1}/{args.n_runs}")
        results = predictor.run_evaluation(graph, edges, run)
        all_results.append(results)

        # Print current results
        print("AUC Scores:")
        print(f"New Temporal: {results['new_temporal']['auc']:.4f} (time: {results['new_temporal']['time']:.2f}s)")
        print(f"Old Temporal: {results['old_temporal']['auc']:.4f} (time: {results['old_temporal']['time']:.2f}s)")
        print(f"Node2Vec: {results['node2vec']['auc']:.4f} (time: {results['node2vec']['time']:.2f}s)")

    # Organize raw results
    raw_results = {
        'params': params,
        'metrics': {
            'new_temporal': {
                'auc_scores': [r['new_temporal']['auc'] for r in all_results],
                'walk_times': [r['new_temporal']['time'] for r in all_results]
            },
            'old_temporal': {
                'auc_scores': [r['old_temporal']['auc'] for r in all_results],
                'walk_times': [r['old_temporal']['time'] for r in all_results]
            },
            'node2vec': {
                'auc_scores': [r['node2vec']['auc'] for r in all_results],
                'walk_times': [r['node2vec']['time'] for r in all_results]
            }
        }
    }


    node2vec_type = 'weighted' if args.weighted_node2vec else 'unweighted'
    directed_suffix = 'directed' if args.directed else 'undirected'
    # Save raw results
    with open(f'save/{args.dataset}_{args.walk_bias}_{args.initial_edge_bias}_{context_window}_{node2vec_type}_{args.edge_operator}_{directed_suffix}.pkl', 'wb') as f:
        pickle.dump(raw_results, f)

    # Print summary statistics
    print("\nFinal Results:")
    for method in ['new_temporal', 'old_temporal', 'node2vec']:
        auc_scores = raw_results['metrics'][method]['auc_scores']
        walk_times = raw_results['metrics'][method]['walk_times']

        print(f"{method}:")
        print(f"  AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
        print(f"  Walk Time: {np.mean(walk_times):.2f}s ± {np.std(walk_times):.2f}s")


if __name__ == "__main__":
    main()