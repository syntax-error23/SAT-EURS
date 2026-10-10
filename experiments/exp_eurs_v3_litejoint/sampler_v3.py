"""
Class-Aware Scene Sampler for EURS-Net V3
Balances rare organism exposure by weighting scenes containing:
- Elasmobranchii (Class 3): 'ray1'
- Other marine organisms (Class 4): 'starfish2', 'turtles1'
Strictly preserves len(sampler) == len(dataset) to guarantee consistent epoch step semantics.
"""

from typing import Iterator, List, Dict
import numpy as np
from torch.utils.data import Sampler


class ClassAwareSceneSampler(Sampler[int]):
    """
    Class-Aware Scene Sampler:
    1. Groups dataset sample indices by scene_id.
    2. Assigns boosted sampling weights to rare-class scenes:
       - 'ray1' (Class 3 Shark/Ray): weight = 3.0
       - 'starfish2' (Class 4 Other): weight = 2.5
       - 'turtles1' (Class 4 Other): weight = 2.5
       - Standard fish scenes (Classes 0, 1, 2): weight = 1.0 each
    3. Samples scenes according to normalized scene probabilities.
    4. Uniformly samples an image variation within the chosen scene.
    5. Yields exactly len(dataset) indices per epoch.
    """
    def __init__(
        self,
        dataset,
        seed: int = 42,
        rare_scene_boosts: Dict[str, float] = None
    ):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        # Group indices by scene_id
        self.scene_to_indices: Dict[str, List[int]] = {}
        for idx in range(len(dataset)):
            sample = dataset.samples[idx]
            scene_id = sample.get("scene_id", f"scene_{idx}")
            if scene_id not in self.scene_to_indices:
                self.scene_to_indices[scene_id] = []
            self.scene_to_indices[scene_id].append(idx)

        self.scene_ids = sorted(list(self.scene_to_indices.keys()))
        assert len(self.scene_ids) > 0, "No scenes found in dataset for ClassAwareSceneSampler!"

        # Default rare boosts
        boosts = {
            'ray1': 3.0,
            'starfish2': 2.5,
            'turtles1': 2.5
        }
        if rare_scene_boosts is not None:
            boosts.update(rare_scene_boosts)

        # Compute sampling probability per scene
        weights = np.array([boosts.get(s, 1.0) for s in self.scene_ids], dtype=np.float64)
        self.scene_probs = weights / weights.sum()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.RandomState(self.seed + self.epoch)
        n_total = len(self.dataset)
        sampled_indices: List[int] = []

        chosen_scenes = rng.choice(self.scene_ids, size=n_total, p=self.scene_probs)
        for scene in chosen_scenes:
            variant_indices = self.scene_to_indices[scene]
            chosen_idx = rng.choice(variant_indices)
            sampled_indices.append(int(chosen_idx))

        return iter(sampled_indices)

    def __len__(self) -> int:
        return len(self.dataset)
