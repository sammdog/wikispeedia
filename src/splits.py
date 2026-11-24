import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_target_splits(
    paths: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """Делит партии на train / val / test строго по целевой статье (target)."""
    unique_targets = np.sort(paths["target"].dropna().unique())

    holdout_size = val_size + test_size
    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
    tr_idx, tmp_idx = next(gss1.split(unique_targets, groups=unique_targets))

    train_targets = set(unique_targets[tr_idx])
    tmp_targets = unique_targets[tmp_idx]

    rel_test = test_size / holdout_size
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    val_rel, te_rel = next(gss2.split(tmp_targets, groups=tmp_targets))

    val_targets = set(tmp_targets[val_rel])
    test_targets = set(tmp_targets[te_rel])

    assert not (train_targets & val_targets)
    assert not (train_targets & test_targets)
    assert not (val_targets & test_targets)

    mapping = {}
    for t in train_targets:
        mapping[t] = "train"
    for t in val_targets:
        mapping[t] = "val"
    for t in test_targets:
        mapping[t] = "test"

    return paths["target"].map(mapping)

