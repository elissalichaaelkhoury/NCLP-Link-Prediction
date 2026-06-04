import argparse
import random
from typing import List, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

from nclp_ga import (
    compute_centralities,
    evaluate_auc,
    run_genetic_algorithm,
)


Edge = Tuple[int, int]


def load_graph(
    file_path: str,
    weighted: bool = False,
    delimiter: str = "auto",
    comment_prefixes=("#", "%"),
) -> nx.Graph:
    """
    Generic graph loader.

    Supports:
        space-separated files:
            source target
            source target weight

        comma-separated files:
            source,target
            source,target,weight

    Parameters:
        file_path: path to the dataset file
        weighted: whether to read the third column as edge weight
        delimiter: "auto", "space", or "comma"
    """
    G = nx.Graph()

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if any(line.startswith(prefix) for prefix in comment_prefixes):
                continue

            if delimiter == "comma":
                parts = line.split(",")
            elif delimiter == "space":
                parts = line.split()
            else:
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()

            if len(parts) < 2:
                continue

            try:
                source = int(parts[0])
                target = int(parts[1])

                if source == target:
                    continue

                if weighted and len(parts) >= 3:
                    weight = float(parts[2])
                    G.add_edge(source, target, weight=weight)
                else:
                    G.add_edge(source, target)

            except ValueError:
                continue

    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def normalize_edge(u: int, v: int) -> Edge:
    """Normalize undirected edge ordering."""
    return (u, v) if u <= v else (v, u)


def get_edges(G: nx.Graph) -> List[Edge]:
    """Return normalized edge list."""
    return [normalize_edge(u, v) for u, v in G.edges()]


def build_graph_from_edges(nodes: List[int], edges: List[Edge]) -> nx.Graph:
    """Build graph while keeping all original nodes."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def split_edges(
    G: nx.Graph,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[nx.Graph, List[Edge], List[Edge]]:
    """
    Split observed edges into train, validation, and test edges.

    Returns:
        train_graph, val_edges, test_edges
    """
    rng = random.Random(seed)

    edges = get_edges(G)
    rng.shuffle(edges)

    n_edges = len(edges)
    n_test = max(1, int(test_ratio * n_edges))
    n_val = max(1, int(val_ratio * n_edges))

    test_edges = edges[:n_test]
    val_edges = edges[n_test:n_test + n_val]
    train_edges = edges[n_test + n_val:]

    train_graph = build_graph_from_edges(list(G.nodes()), train_edges)

    return train_graph, val_edges, test_edges


def sample_negative_edges(
    G_original: nx.Graph,
    num_samples: int,
    seed: int = 42,
) -> List[Edge]:
    """
    Sample negative edges from pairs that are not connected in the original graph.
    """
    rng = random.Random(seed)

    nodes = list(G_original.nodes())
    existing_edges = set(get_edges(G_original))

    negative_edges = set()
    max_attempts = num_samples * 100
    attempts = 0

    while len(negative_edges) < num_samples and attempts < max_attempts:
        u, v = rng.sample(nodes, 2)
        edge = normalize_edge(u, v)

        if edge not in existing_edges:
            negative_edges.add(edge)

        attempts += 1

    if len(negative_edges) < num_samples:
        all_non_edges = [
            normalize_edge(u, v)
            for u, v in nx.non_edges(G_original)
        ]
        rng.shuffle(all_non_edges)
        negative_edges = set(all_non_edges[:num_samples])

    return list(negative_edges)


def run_single_experiment(args, run_seed: int):
    """
    Run one train/validation/test experiment.
    """
    G = load_graph(
        args.file_path,
        weighted=args.weighted,
        delimiter=args.delimiter,
    )

    print("\nLoaded graph")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    train_graph, val_pos_edges, test_pos_edges = split_edges(
        G,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=run_seed,
    )

    val_neg_edges = sample_negative_edges(
        G,
        num_samples=len(val_pos_edges),
        seed=run_seed + 1000,
    )

    test_neg_edges = sample_negative_edges(
        G,
        num_samples=len(test_pos_edges),
        seed=run_seed + 2000,
    )

    print("\nSplit")
    print(f"Training edges: {train_graph.number_of_edges()}")
    print(f"Validation positive edges: {len(val_pos_edges)}")
    print(f"Validation negative edges: {len(val_neg_edges)}")
    print(f"Test positive edges: {len(test_pos_edges)}")
    print(f"Test negative edges: {len(test_neg_edges)}")

    best_weights, best_val_auc = run_genetic_algorithm(
        train_graph=train_graph,
        val_positive_edges=val_pos_edges,
        val_negative_edges=val_neg_edges,
        generations=args.generations,
        population_size=args.population_size,
        initial_mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elitism_rate=args.elitism_rate,
        katz_alpha=args.katz_alpha,
        seed=run_seed,
    )

    centralities = compute_centralities(train_graph, katz_alpha=args.katz_alpha)

    test_auc = evaluate_auc(
        train_graph,
        test_pos_edges,
        test_neg_edges,
        best_weights,
        centralities,
    )

    print("\nFinal result")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Best weights: {[round(w, 6) for w in best_weights]}")

    return {
        "seed": run_seed,
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "weights": best_weights,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generic NCLP + GA link prediction pipeline"
    )

    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to edge-list file",
    )

    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use third column as edge weight if available",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default="auto",
        choices=["auto", "space", "comma"],
        help="Dataset delimiter",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of repeated experiments",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of edges used for testing",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of edges used for validation during GA",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of GA generations",
    )

    parser.add_argument(
        "--population_size",
        type=int,
        default=50,
        help="GA population size",
    )

    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.5,
        help="Initial mutation rate",
    )

    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=0.8,
        help="Crossover probability",
    )

    parser.add_argument(
        "--elitism_rate",
        type=float,
        default=0.1,
        help="Fraction of elite individuals retained",
    )

    parser.add_argument(
        "--katz_alpha",
        type=float,
        default=0.01,
        help="Katz attenuation parameter",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    results = []

    for i in range(args.runs):
        run_seed = args.seed + i
        print("\n" + "=" * 70)
        print(f"Run {i + 1}/{args.runs} | seed={run_seed}")
        print("=" * 70)

        result = run_single_experiment(args, run_seed)
        results.append(result)

    test_aucs = [r["test_auc"] for r in results]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean Test AUC: {np.mean(test_aucs):.4f}")
    print(f"Std Test AUC:  {np.std(test_aucs):.4f}")

    all_weights = np.array([r["weights"] for r in results])
    avg_weights = np.mean(all_weights, axis=0)

    print(f"Average weights: {[round(w, 6) for w in avg_weights]}")
    print("=" * 70)


if __name__ == "__main__":
    main()