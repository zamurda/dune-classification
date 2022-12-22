"""
Train-test splitting which preserves indices.
"""
import numpy as np

__all__ = [
    "train_test_split"
]


def _generate_rand():
    generator = np.random.default_rng()
    return generator.integers(np.iinfo(np.int64).max)


def train_test_split(features, targets, test_ratio=0.25, return_index=True, random_state=None):
    """
    custom train-test splitting which has virtually same implementation of sklearn.train_test_split.


    """

    rng = np.random.default_rng(seed=int(random_state)) if random_state is not None\
        else np.random.default_rng(seed=_generate_rand())

    if len(features) != len(targets):

        raise ValueError(
            "features and targets must have the same length"
        )

    else:

        index_array = np.array(range(len(targets)))

        together = (np.vstack([index_array, features.transpose(), targets])).transpose()
        rng.shuffle(together)

        targets = together[:, -1]
        index_array = together[:, 0]
        features = together[:, 1:-1]

        split = int(len(targets) * (1-test_ratio))  # index at which train-test is split up

        return (
            features[:split],
            features[split:],
            targets[:split],
            targets[split:]
        ) if not return_index else (
            features[:split],
            features[split:],
            targets[:split],
            targets[split:],
            index_array[:split],
            index_array[split:]
            )

