import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DIR
from src.data_loader import load_article_text, load_articles, load_links, load_paths
from src.graph_utils import build_graph, load_distance_matrix
from src.splits import assign_target_splits


def compute_train_drop_rate(paths: pd.DataFrame, min_visits: int = 10) -> dict[str, float]:
    train_df = paths[paths["split"] == "train"]
    visits = Counter()
    drops = Counter()

    for trace, finished in zip(train_df["trace"], train_df["finished"]):
        if len(trace) < 2:
            continue
        if finished == 1:
            visits.update(trace[1:-1])
        else:
            visits.update(trace[1:])
            drops[trace[-1]] += 1

    return {
        art: (drops[art] / cnt if cnt >= min_visits else 0.0)
        for art, cnt in visits.items()
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1/5 Загрузка статей, ссылок и партий...")
    articles = load_articles()
    links = load_links()
    paths = load_paths()

    print("2/5 Разбиение train / val / test по target...")
    paths["split"] = assign_target_splits(paths, val_size=0.15, test_size=0.15, seed=42)
    print(paths["split"].value_counts().to_string())

    print("3/5 Построение графа, степеней, PageRank и drop_rate (только по train)...")
    G = build_graph(links, articles)
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85)
    drop_rate = compute_train_drop_rate(paths, min_visits=10)

    node_features = articles.copy()
    node_features["in_degree"] = node_features["article"].map(in_deg).fillna(0).astype(int)
    node_features["out_degree"] = node_features["article"].map(out_deg).fillna(0).astype(int)
    node_features["pagerank"] = node_features["article"].map(pagerank).fillna(0.0)
    node_features["drop_rate_train"] = node_features["article"].map(drop_rate).fillna(0.0)

    print("4/5 Загрузка матрицы кратчайших расстояний D...")
    D, article_to_idx = load_distance_matrix(articles)

    print("5/5 Расчёт матрицы косинусной близости TF-IDF (4604x4604)...")
    texts = [load_article_text(raw_name) for raw_name in articles["article_raw"]]
    if any(t.strip() for t in texts):
        vec = TfidfVectorizer(stop_words="english", max_features=30000, dtype=np.float32)
        tfidf_mat = vec.fit_transform(texts)
        # Строки TF-IDF уже L2-нормированы, поэтому скалярное произведение = косинусная близость
        sim_matrix = (tfidf_mat @ tfidf_mat.T).toarray().astype(np.float32)
    else:
        print("  [инфо] Тексты статей не найдены в plaintext_articles/. Инициализирована единичная матрица.")
        n = len(articles)
        sim_matrix = np.eye(n, dtype=np.float32)

    paths.to_parquet(PROCESSED_DIR / "paths.parquet", index=False)
    node_features.to_parquet(PROCESSED_DIR / "node_features.parquet", index=False)
    np.save(PROCESSED_DIR / "distance_matrix.npy", D)
    np.save(PROCESSED_DIR / "tfidf_sim.npy", sim_matrix)
    with open(PROCESSED_DIR / "article_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(article_to_idx, f, ensure_ascii=False)

    print("\nГотово! Артефакты сохранены в:", PROCESSED_DIR)


if __name__ == "__main__":
    main()

