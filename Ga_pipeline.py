import math
import random
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score


Edge = Tuple[int, int]


def normalize_weights(weights: List[float]) -> List[float]:
    """Clip negative values and normalize weights to sum to 1."""
    weights = [max(0.0, float(w)) for w in weights]
    total = sum(weights)

    if total == 0:
        return [1.0 / len(weights)] * len(weights)

    return [w / total for w in weights]


def minmax_normalize(values: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize centrality values."""
    if not values:
        return values

    min_val = min(values.values())
    max_val = max(values.values())

    if max_val == min_val:
        return {node: 0.0 for node in values}

    return {
        node: (value - min_val) / (max_val - min_val)
        for node, value in values.items()
    }


def compute_centralities(G: nx.Graph, katz_alpha: float = 0.01) -> Dict[str, Dict[int, float]]:
    """
    Compute and normalize betweenness, closeness, and Katz centralities.
    """
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)

    try:
        katz = nx.katz_centrality(
            G,
            alpha=katz_alpha,
            beta=1.0,
            max_iter=50000,
            tol=1e-06,
            normalized=True,
            weight=None,
        )
    except nx.PowerIterationFailedConvergence:
        print("Warning: Katz centrality did not converge. Using zeros.")
        katz = {node: 0.0 for node in G.nodes()}

    return {
        "betweenness": minmax_normalize(betweenness),
        "closeness": minmax_normalize(closeness),
        "katz": minmax_normalize(katz),
    }


def safe_shortest_path_length(G: nx.Graph, u: int, v: int) -> int:
    """
    Return shortest path length. If disconnected, return number of nodes.
    """
    try:
        return nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        return G.number_of_nodes()


def resource_allocation(G: nx.Graph, u: int, v: int) -> float:
    """Resource Allocation score."""
    common_neighbors = nx.common_neighbors(G, u, v)
    score = 0.0

    for z in common_neighbors:
        degree_z = G.degree(z)
        if degree_z > 0:
            score += 1.0 / degree_z

    return score


def adamic_adar(G: nx.Graph, u: int, v: int) -> float:
    """Adamic-Adar score."""
    common_neighbors = nx.common_neighbors(G, u, v)
    score = 0.0

    for z in common_neighbors:
        degree_z = G.degree(z)
        if degree_z > 1:
            score += 1.0 / math.log(degree_z)

    return score


def nclp_score(
    G: nx.Graph,
    u: int,
    v: int,
    weights: List[float],
    centralities: Dict[str, Dict[int, float]],
) -> float:
    """
    Compute NCLP score for one candidate pair (u, v).

    weights:
        w1, w2, w3: betweenness, closeness, Katz for node u
        w4, w5, w6: betweenness, closeness, Katz for node v
    """
    w1, w2, w3, w4, w5, w6 = normalize_weights(weights)

    cb = centralities["betweenness"]
    cc = centralities["closeness"]
    ck = centralities["katz"]

    distance = safe_shortest_path_length(G, u, v)

    node_term = (
        (w1 * cb.get(u, 0.0) + w4 * cb.get(v, 0.0))
        + (w2 * cc.get(u, 0.0) + w5 * cc.get(v, 0.0))
        + (w3 * ck.get(u, 0.0) + w6 * ck.get(v, 0.0))
    ) / (distance + 1)

    ra = resource_allocation(G, u, v)
    aa = adamic_adar(G, u, v)

    cn_term = 0.0
    for z in nx.common_neighbors(G, u, v):
        cn_term += cc.get(z, 0.0) + cb.get(z, 0.0)

    return node_term + ra + aa + cn_term


def score_pairs(
    G: nx.Graph,
    pairs: List[Edge],
    weights: List[float],
    centralities: Dict[str, Dict[int, float]],
) -> List[float]:
    """Compute NCLP scores for a list of node pairs."""
    return [
        nclp_score(G, u, v, weights, centralities)
        for u, v in pairs
    ]


def evaluate_auc(
    G: nx.Graph,
    positive_edges: List[Edge],
    negative_edges: List[Edge],
    weights: List[float],
    centralities: Dict[str, Dict[int, float]],
) -> float:
    """Compute AUC using positive and negative edge pairs."""
    pairs = positive_edges + negative_edges
    labels = [1] * len(positive_edges) + [0] * len(negative_edges)

    scores = score_pairs(G, pairs, weights, centralities)

    return roc_auc_score(labels, scores)


def tournament_selection(
    population: List[List[float]],
    scores: List[float],
    tournament_size: int = 3,
) -> List[float]:
    """Tournament selection."""
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = max(selected_indices, key=lambda i: scores[i])
    return population[best_index]


def crossover(
    parent1: List[float],
    parent2: List[float],
    crossover_rate: float,
) -> Tuple[List[float], List[float]]:
    """Uniform crossover."""
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]

    child1 = []
    child2 = []

    for p1, p2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(p1)
            child2.append(p2)
        else:
            child1.append(p2)
            child2.append(p1)

    return normalize_weights(child1), normalize_weights(child2)


def mutate(
    weights: List[float],
    mutation_rate: float,
    generation: int,
    max_generations: int,
) -> List[float]:
    """Adaptive mutation with non-negative clipping."""
    if random.random() > mutation_rate:
        return weights

    perturbation_size = 1.0 - (generation / max_generations)

    mutated = [
        w + random.uniform(-perturbation_size, perturbation_size)
        for w in weights
    ]

    return normalize_weights(mutated)


def run_genetic_algorithm(
    train_graph: nx.Graph,
    val_positive_edges: List[Edge],
    val_negative_edges: List[Edge],
    generations: int = 50,
    population_size: int = 50,
    initial_mutation_rate: float = 0.5,
    crossover_rate: float = 0.8,
    elitism_rate: float = 0.1,
    katz_alpha: float = 0.01,
    seed: int = 42,
) -> Tuple[List[float], float]:
    """
    Run GA to optimize the six NCLP weights.

    Returns:
        best_weights, best_validation_auc
    """
    random.seed(seed)
    np.random.seed(seed)

    centralities = compute_centralities(train_graph, katz_alpha=katz_alpha)

    def fitness(weights: List[float]) -> float:
        weights = normalize_weights(weights)
        return evaluate_auc(
            train_graph,
            val_positive_edges,
            val_negative_edges,
            weights,
            centralities,
        )

    population = [
        normalize_weights(np.random.rand(6).tolist())
        for _ in range(population_size)
    ]

    best_weights = None
    best_auc = float("-inf")

    mutation_rate = initial_mutation_rate
    no_improvement_count = 0
    last_best_auc = float("-inf")

    elite_count = max(1, int(population_size * elitism_rate))

    for generation in range(generations):
        scores = [fitness(individual) for individual in population]

        current_best_index = int(np.argmax(scores))
        current_best_auc = scores[current_best_index]
        current_best_weights = population[current_best_index]

        if current_best_auc > best_auc:
            best_auc = current_best_auc
            best_weights = current_best_weights

        sorted_population = [
            individual for _, individual in sorted(
                zip(scores, population),
                key=lambda x: x[0],
                reverse=True,
            )
        ]

        elites = sorted_population[:elite_count]

        new_population = elites[:]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)

            child1, child2 = crossover(parent1, parent2, crossover_rate)

            child1 = mutate(child1, mutation_rate, generation, generations)
            child2 = mutate(child2, mutation_rate, generation, generations)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

        mutation_rate = min(0.5, max(0.1, mutation_rate - 0.01))

        if abs(current_best_auc - last_best_auc) < 1e-8:
            no_improvement_count += 1
            if no_improvement_count > 5:
                mutation_rate = min(0.5, mutation_rate + 0.05)
                no_improvement_count = 0
        else:
            no_improvement_count = 0

        last_best_auc = current_best_auc

        print(
            f"Generation {generation + 1}/{generations} | "
            f"Best validation AUC: {best_auc:.4f} | "
            f"Weights: {[round(w, 4) for w in best_weights]}"
        )

    return normalize_weights(best_weights), best_auc