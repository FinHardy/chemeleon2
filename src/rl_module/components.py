"""Reward components for reinforcement learning."""

from abc import ABC, abstractmethod
from collections import defaultdict

import amd
import torch
from pymatgen.core import Structure

from src.utils.featurizer import featurize
from src.utils.metrics import Metrics, structures_to_amd

###############################################################################
#                              Reward Components                              #
###############################################################################


class RewardComponent(ABC, torch.nn.Module):
    """Base class for all reward components."""

    required_metrics: list[str] = []

    def __init__(
        self,
        weight: float = 1.0,
        normalize_fn: str | None = None,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.weight = weight
        self.normalize_fn = normalize_fn
        self.eps = eps

    @abstractmethod
    def compute(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the raw reward."""
        pass

    def forward(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """Compute and optionally normalize the reward."""
        rewards = self.compute(
            **kwargs,
        )
        if self.normalize_fn:
            rewards = self._normalize(rewards)

        if "device" in kwargs:
            rewards = rewards.to(kwargs["device"])

        return rewards * self.weight

    def _normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.normalize_fn == "norm":
            rewards = normalize(rewards, eps=self.eps)
        elif self.normalize_fn == "std":
            rewards = standardize(rewards, eps=self.eps)
        elif self.normalize_fn == "subtract_mean":
            rewards = rewards - rewards.mean()
        elif self.normalize_fn == "clip":
            rewards = torch.clamp(rewards, min=-1.0, max=1.0)
        elif self.normalize_fn is None:
            pass
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalize_fn}. Use 'norm', 'std', 'clip', or None."
            )
        return rewards


###############################################################################
#                           Custom Reward Components                          #
###############################################################################
class CustomReward(RewardComponent):
    """Wrapper for user-defined custom reward functions."""

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        """Placeholder for custom reward function."""
        return torch.zeros(len(gen_structures))


###############################################################################
#                          Built-in Reward Components                         #
###############################################################################
class CreativityReward(RewardComponent):
    """Combined Unique and Novel reward with AMD distance fallback."""

    required_metrics = ["unique", "novel"]

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        **kwargs,
    ) -> torch.Tensor:
        # Ensure metrics_obj is provided in kwargs
        reference_structures = metrics_obj._reference_structures
        metrics_results = metrics_obj._results

        # Build a mapping from formula to reference structures
        ref_structures_by_formula = defaultdict(list)
        for ref_structure in reference_structures + gen_structures:
            ref_structures_by_formula[ref_structure.reduced_formula].append(
                ref_structure
            )

        # Compute creativity rewards
        rewards = []
        for i, gen_structure in enumerate(gen_structures):
            u, v = metrics_results["unique"][i], metrics_results["novel"][i]
            if u and v:
                r = 1.0
            elif not u and not v:
                r = 0.0
            else:
                matching_refs = ref_structures_by_formula.get(
                    gen_structure.reduced_formula, []
                )
                amds = structures_to_amd([gen_structure] + matching_refs, 100)
                dists = amd.AMD_cdist(amds, amds)[0]
                r = dists[dists > 0].min()
            rewards.append(r)

        return torch.as_tensor(rewards).float()


class EnergyReward(RewardComponent):
    """Rewards structures with low energy above convex hull."""

    required_metrics = ["e_above_hull"]

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        **kwargs,
    ) -> torch.Tensor:
        metrics_results = metrics_obj._results

        r_energy = torch.as_tensor(metrics_results["e_above_hull"]).float()
        r_energy = r_energy.nan_to_num(nan=1.0)  # max clip energy
        r_energy = r_energy.clamp(min=0.0, max=1.0)
        r_energy = r_energy * -1.0  # Negative for minimization
        return r_energy


class StructureDiversityReward(RewardComponent):
    """Rewards diverse crystal structures using MMD."""

    required_metrics = ["structure_diversity"]

    def __init__(self, max_num_reference: int = 50000, **kwargs):
        super().__init__(**kwargs)
        self.max_num_reference = max_num_reference
        self._cached_ref_features = None

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        # Cache reference features on first call (on CPU)
        if self._cached_ref_features is None:
            ref_structure_features = metrics_obj._reference_structure_features
            assert ref_structure_features is not None

            if len(ref_structure_features) > self.max_num_reference:
                indices = torch.randperm(len(ref_structure_features))[
                    : self.max_num_reference
                ]
                self._cached_ref_features = ref_structure_features[indices]
            else:
                self._cached_ref_features = ref_structure_features

        # Compute on CPU
        gen_features = featurize(gen_structures)
        gen_structure_features = gen_features["structure_features"]

        # MMD computation on CPU (no GPU memory needed)
        r_structure_diversity = mmd_reward(
            z_gen=gen_structure_features, z_ref=self._cached_ref_features
        )["r_indiv"]

        # Move result to target device
        return r_structure_diversity.to(device)


class CompositionDiversityReward(RewardComponent):
    """Rewards diverse chemical compositions using MMD."""

    required_metrics = ["composition_diversity"]

    def __init__(self, max_num_reference: int = 50000, **kwargs):
        super().__init__(**kwargs)
        self.max_num_reference = max_num_reference
        self._cached_ref_features = None

    def compute(
        self,
        gen_structures: list[Structure],
        metrics_obj: Metrics,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        # Cache reference features on first call (on CPU)
        if self._cached_ref_features is None:
            ref_composition_features = metrics_obj._reference_composition_features
            assert ref_composition_features is not None

            if len(ref_composition_features) > self.max_num_reference:
                indices = torch.randperm(len(ref_composition_features))[
                    : self.max_num_reference
                ]
                self._cached_ref_features = ref_composition_features[indices]
            else:
                self._cached_ref_features = ref_composition_features

        # Compute on CPU
        gen_features = featurize(gen_structures)
        gen_composition_features = gen_features["composition_features"]

        # MMD computation on CPU (no GPU memory needed)
        r_composition_diversity = mmd_reward(
            z_gen=gen_composition_features, z_ref=self._cached_ref_features
        )["r_indiv"]

        # Move result to target device
        return r_composition_diversity.to(device)


###############################################################################
#                                  Utils                                      #
###############################################################################


def standardize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.std() < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std() + eps)


def normalize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if x.max() - x.min() < eps:
        return torch.zeros_like(x)
    x = (x - x.min()) / (x.max() - x.min() + eps)
    return x.clamp(0.0, 1.0)


def mmd_reward(z_gen, z_ref, r_term=None):
    """Training Diffusion Models Towards Diverse Image Generation with Reinforcement Learning.

    Args:
        z_gen: Generated sample features.
        z_ref: Reference sample features.
        r_term: Pre-computed R_term scalar. If provided, skips k_rr computation.
    """

    def poly_k(z, y, deg=3):
        d = z.size(-1)
        return (z @ y.T / d + 1) ** deg

    M, N = len(z_gen), len(z_ref)

    k_gg = poly_k(z_gen, z_gen)
    k_gr = poly_k(z_gen, z_ref)

    # Compute R_term (skip if pre-computed)
    if r_term is None:
        k_rr = poly_k(z_ref, z_ref)
        R_term = (k_rr.sum() - k_rr.trace()) / (N * (N - 1))
    else:
        R_term = r_term
    G = k_gg.sum() - k_gg.trace()
    C = k_gr.sum()
    mmd_full = G / (M * (M - 1)) + R_term - 2 * C / (M * N)  # Eq. (11)

    # Compute individual MMD
    S = k_gg.sum(dim=1) - k_gg.diagonal()  # S_m
    T = k_gr.sum(dim=1)  # T_m

    Mp = M - 1
    Ap = Mp * (Mp - 1)
    mmd_drop = (G - 2 * S) / Ap + R_term - 2 * (C - T) / (Mp * N)

    r_indiv = mmd_drop - mmd_full
    return {"r": -mmd_full, "r_indiv": r_indiv}
