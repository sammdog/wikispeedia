from urllib.parse import unquote
import pandas as pd
from src.config import GRAPH_DIR, PLAINTEXT_DIR


def load_articles() -> pd.DataFrame:
    df = pd.read_csv(
        GRAPH_DIR / "articles.tsv",
        sep="\t",
        header=None,
        names=["article_raw"],
        comment="#",
        quoting=3,
    )
    df["article"] = df["article_raw"].map(unquote)
    return df


def load_links() -> pd.DataFrame:
    df = pd.read_csv(
        GRAPH_DIR / "links.tsv",
        sep="\t",
        header=None,
        names=["src", "tgt"],
        comment="#",
        quoting=3,
    )
    df["src"] = df["src"].map(unquote)
    df["tgt"] = df["tgt"].map(unquote)
    return df


def load_categories() -> pd.DataFrame:
    df = pd.read_csv(
        GRAPH_DIR / "categories.tsv",
        sep="\t",
        header=None,
        names=["article", "category"],
        comment="#",
        quoting=3,
    )
    df["article"] = df["article"].map(unquote)
    df["category"] = df["category"].map(unquote)
    return df


def parse_path(path_str: str) -> tuple[list[str], list[str], list[bool]]:
    """
    Возвращает (trace, effective_path, is_back):
      - trace: реальная последовательность статей с учётом возвратов (длина - 1 = число шагов k)
      - effective_path: путь без откатов назад
      - is_back: флаг той же длины, что и trace (True, если на эту статью вернулись шагом назад)
    """
    tokens = path_str.split(";")
    trace: list[str] = []
    stack: list[str] = []
    is_back: list[bool] = []

    for tok in tokens:
        if tok == "<":
            if len(stack) > 1:
                stack.pop()
            if stack:
                trace.append(stack[-1])
                is_back.append(True)
        else:
            name = unquote(tok)
            stack.append(name)
            trace.append(name)
            is_back.append(False)

    return trace, stack, is_back


def load_paths() -> pd.DataFrame:
    fin = pd.read_csv(
        GRAPH_DIR / "paths_finished.tsv",
        sep="\t",
        header=None,
        names=["ip", "timestamp", "duration", "path", "rating"],
        comment="#",
        quoting=3,
    )
    unf = pd.read_csv(
        GRAPH_DIR / "paths_unfinished.tsv",
        sep="\t",
        header=None,
        names=["ip", "timestamp", "duration", "path", "target", "type"],
        comment="#",
        quoting=3,
    )

    fin_parsed = fin["path"].map(parse_path)
    fin["trace"], fin["effective_path"], fin["is_back"] = zip(*fin_parsed)
    fin["start"] = fin["trace"].map(lambda s: s[0] if s else None)
    fin["target"] = fin["trace"].map(lambda s: s[-1] if s else None)
    fin["finished"] = 1
    fin["type"] = "finished"

    unf_parsed = unf["path"].map(parse_path)
    unf["trace"], unf["effective_path"], unf["is_back"] = zip(*unf_parsed)
    unf["start"] = unf["trace"].map(lambda s: s[0] if s else None)
    unf["target"] = unf["target"].map(unquote)  # исправлен баг с отсутствием unquote
    unf["finished"] = 0
    unf["rating"] = pd.NA

    cols = [
        "ip", "timestamp", "duration", "start", "target",
        "finished", "type", "rating", "trace", "effective_path", "is_back",
    ]
    paths = pd.concat([fin[cols], unf[cols]], ignore_index=True)
    paths["n_steps"] = paths["trace"].map(lambda s: len(s) - 1)
    paths["n_back_clicks"] = paths["is_back"].map(sum)
    paths.insert(0, "game_id", range(len(paths)))
    return paths


def load_article_text(article_raw: str) -> str:
    """Читает текст статьи по исходному URL-encoded имени файла из plaintext_articles/."""
    file_path = PLAINTEXT_DIR / f"{article_raw}.txt"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8", errors="ignore")
    return ""

