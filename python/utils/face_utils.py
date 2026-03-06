import numpy as np


def compute_similarity(feat1: np.ndarray, feat2) -> float:
    """
    余弦相似度，值域 [-1, 1]，越大越相似。
    feat1/feat2: 任意形状，自动展平。
    """
    if not isinstance(feat2, np.ndarray):
        feat2 = np.array(feat2)
    f1 = feat1.ravel().astype(np.float32)
    f2 = feat2.ravel().astype(np.float32)
    return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))
