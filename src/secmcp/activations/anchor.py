from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureBundle:
    features: object
    anchors: object


def flatten_activations(activations, mode: str = "concat"):
    if mode == "concat":
        return activations.flatten(1).float()
    if mode == "last_only":
        return activations[:, -1, :].float()
    raise ValueError(f"Unknown layer feature mode: {mode}")


def select_anchors(train_activations, train_labels, n_anchors: int, seed: int = 42):
    import torch

    benign_idx = (train_labels == 0).nonzero(as_tuple=False).flatten()
    if benign_idx.numel() == 0:
        raise ValueError("Cannot select anchors: no benign training samples")
    n = min(int(n_anchors), int(benign_idx.numel()))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = benign_idx[torch.randperm(benign_idx.numel(), generator=generator)[:n]]
    return train_activations[selected].clone()


def distance_features(samples, anchors, mode: str = "concat", chunk_size: int = 256):
    import torch

    sample_flat = flatten_activations(samples, mode=mode)
    anchor_flat = flatten_activations(anchors, mode=mode)
    chunks = []
    for start in range(0, sample_flat.shape[0], chunk_size):
        chunk = sample_flat[start : start + chunk_size]
        dists = torch.cdist(chunk, anchor_flat)
        chunks.append(
            torch.stack(
                [
                    dists.mean(dim=1),
                    dists.min(dim=1).values,
                    dists.max(dim=1).values,
                    dists.std(dim=1),
                ],
                dim=1,
            )
        )
    return torch.cat(chunks, dim=0)


def diff_features(samples, anchors, mode: str = "concat"):
    anchor_mean = anchors.mean(dim=0, keepdim=True)
    return flatten_activations(samples - anchor_mean, mode=mode)
