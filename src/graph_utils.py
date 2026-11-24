import networkx as nx
import numpy as np
import pandas as pd
from src.config import GRAPH_DIR


def build_graph(links: pd.DataFrame, articles: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(articles["article"].tolist())
    G.add_edges_from(zip(links["src"], links["tgt"]))
    return G


def load_distance_matrix(articles: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    with open(GRAPH_DIR / "shortest-path-distance-matrix.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    n = len(articles)
    D = np.full((n, n), -1, dtype=np.int8)
    for i, line in enumerate(lines):
        for j, ch in enumerate(line):
            D[i, j] = -1 if ch == "_" else int(ch)

    article_to_idx = {name: i for i, name in enumerate(articles["article"])}
    return D, article_to_idx

