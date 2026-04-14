"""Phase 1 ferroelectric reward components for RL training.

These rewards push the diffusion policy toward non-centrosymmetric polar
structures — the necessary structural precondition for ferroelectricity.

All three components work directly on decoded crystal structures using spglib
and pymatgen, with no MLIP or DFT calls required.
"""

import numpy as np
import torch
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure

from src.rl_module.components import RewardComponent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 68 space groups belonging to the 10 polar point groups.
# C1(1), C2(3-5), Cs(6-9), C2v(25-46), C4(75-80), C4v(99-110),
# C3(143-146), C3v(156-161), C6(168-173), C6v(183-186)
POLAR_SPACE_GROUPS: frozenset[int] = frozenset([
    # C1 (point group 1)
    1,
    # C2 (point group 2)
    3, 4, 5,
    # Cs (point group m)
    6, 7, 8, 9,
    # C2v (point group mm2)
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    # C4 (point group 4)
    75, 76, 77, 78, 79, 80,
    # C4v (point group 4mm)
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    # C3 (point group 3)
    143, 144, 145, 146,
    # C3v (point group 3m)
    156, 157, 158, 159, 160, 161,
    # C6 (point group 6)
    168, 169, 170, 171, 172, 173,
    # C6v (point group 6mm)
    183, 184, 185, 186,
])

# d0 transition metals — off-centering driven by second-order Jahn-Teller effect
_FE_B_SITE_ELEMENTS: frozenset[str] = frozenset([
    "Ti", "Zr", "Hf", "Nb", "Ta", "V", "W", "Mo", "Mn", "Cr",
])

# Lone-pair cations — stereochemically active ns² electrons drive off-centering
_LONE_PAIR_CATIONS: frozenset[str] = frozenset([
    "Pb", "Bi", "Sn", "Sb", "Te", "Tl", "In", "Ge",
])

# d0 TMs relevant for FE chemistry scoring (slightly narrower than _FE_B_SITE_ELEMENTS)
_D0_TRANSITION_METALS: frozenset[str] = frozenset([
    "Ti", "Zr", "Hf", "Nb", "Ta", "V", "W", "Mo",
])

# Common anions in oxide/halide/chalcogenide ferroelectrics
_FE_ANIONS: frozenset[str] = frozenset([
    "O", "F", "Cl", "Br", "I", "S", "Se", "Te",
])

# Reference displacement for normalization — ~0.3 Å is a generous upper bound
# (BaTiO₃ Ti off-centering is ~0.12 Å in the room-temperature tetragonal phase)
_DISTORTION_NORM_FACTOR: float = 0.3


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------


class PolarSymmetryReward(RewardComponent):
    """Binary reward: 1.0 if structure belongs to a polar space group, 0.0 otherwise.

    Uses spglib symmetry detection with a loose tolerance suited for
    unrelaxed as-generated crystal structures (symprec=0.2 by default).
    Tighten to 0.05–0.1 after MLFF relaxation in Phase 2.
    """

    metric_key = "fe/polar_hit_rate"

    def __init__(self, symprec: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.symprec = symprec

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        rewards = [_polar_symmetry_score(s, self.symprec) for s in gen_structures]
        return torch.as_tensor(rewards, dtype=torch.float32)


class PolarDistortionReward(RewardComponent):
    """Reward based on cation off-centering from coordination polyhedron centroids.

    For each d0 transition-metal B-site cation, measures its displacement from
    the centroid of its nearest-neighbour coordination polyhedron. The mean
    displacement across all B-sites is normalised to [0, 1].

    For structures without identifiable B-site cations, falls back to a
    bond-length asymmetry index.
    """

    metric_key = "fe/polar_distortion"

    def __init__(self, norm_factor: float = _DISTORTION_NORM_FACTOR, **kwargs):
        super().__init__(**kwargs)
        self.norm_factor = norm_factor

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        rewards = [_polar_distortion_score(s, self.norm_factor) for s in gen_structures]
        return torch.as_tensor(rewards, dtype=torch.float32)


class FEChemistryReward(RewardComponent):
    """Compositional heuristic reward based on known ferroelectric chemistry families.

    Scoring breakdown (max 1.0):
        +0.35  lone-pair cation present (Pb, Bi, Sn, ...)
        +0.35  d0 transition metal present (Ti, Zr, Nb, Ta, ...)
        +0.20  perovskite-like ABO₃/ABX₃ stoichiometry
        +0.10  oxide/halide/chalcogenide anion present
    """

    metric_key = "fe/fe_chemistry"

    def compute(self, gen_structures: list[Structure], **kwargs) -> torch.Tensor:
        rewards = [_fe_chemistry_score(s) for s in gen_structures]
        return torch.as_tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Private scoring functions
# ---------------------------------------------------------------------------


def _polar_symmetry_score(structure: Structure, symprec: float) -> float:
    """Return 1.0 if structure is in a polar space group, else 0.0."""
    try:
        import spglib
        from pymatgen.io.ase import AseAtomsAdaptor

        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            return 0.0
        return 1.0 if dataset["number"] in POLAR_SPACE_GROUPS else 0.0
    except Exception:
        return 0.0


def _polar_distortion_score(structure: Structure, norm_factor: float) -> float:
    """Return normalised mean B-site off-centering displacement."""
    try:
        cnn = CrystalNN()
        displacements = []

        for i, site in enumerate(structure):
            if site.specie.symbol not in _FE_B_SITE_ELEMENTS:
                continue
            try:
                nn_info = cnn.get_nn_info(structure, i)
            except Exception:
                continue
            if len(nn_info) < 4:
                continue  # skip under-coordinated sites

            neighbour_coords = np.array([nn["site"].coords for nn in nn_info])
            centroid = neighbour_coords.mean(axis=0)
            displacement = float(np.linalg.norm(site.coords - centroid))
            displacements.append(displacement)

        if displacements:
            mean_disp = float(np.mean(displacements))
            return float(np.clip(mean_disp / norm_factor, 0.0, 1.0))

        # Fallback: bond-length asymmetry for structures without B-site cations
        return _bond_length_asymmetry_score(structure, cnn)

    except Exception:
        return 0.0


def _bond_length_asymmetry_score(structure: Structure, cnn: CrystalNN | None = None) -> float:
    """Bond-length variance fallback: high variance = more asymmetric bonding."""
    try:
        if cnn is None:
            cnn = CrystalNN()
        bond_lengths = []

        for i, site in enumerate(structure):
            if site.specie.symbol in _FE_ANIONS:
                continue
            try:
                nn_info = cnn.get_nn_info(structure, i)
            except Exception:
                continue
            for nn in nn_info:
                if nn["site"].specie.symbol in _FE_ANIONS:
                    dist = float(np.linalg.norm(site.coords - nn["site"].coords))
                    bond_lengths.append(dist)

        if len(bond_lengths) < 2:
            return 0.0

        bond_arr = np.array(bond_lengths)
        asymmetry = np.std(bond_arr) / (np.mean(bond_arr) + 1e-8)
        # Typical range: 0.0 (perfectly symmetric) to ~0.15 (highly asymmetric)
        return float(np.clip(asymmetry / 0.15, 0.0, 1.0))

    except Exception:
        return 0.0


def _fe_chemistry_score(structure: Structure) -> float:
    """Additive compositional heuristic for ferroelectric-relevant chemistry."""
    try:
        elements = {str(el) for el in structure.composition.elements}
        score = 0.0

        if elements & _LONE_PAIR_CATIONS:
            score += 0.35
        if elements & _D0_TRANSITION_METALS:
            score += 0.35
        if _is_perovskite_like(structure.composition):
            score += 0.20
        if elements & _FE_ANIONS:
            score += 0.10

        return float(np.clip(score, 0.0, 1.0))

    except Exception:
        return 0.0


def _is_perovskite_like(comp: Composition) -> bool:
    """Return True for ABO₃/ABX₃-like stoichiometry (±tolerance)."""
    try:
        reduced = comp.reduced_composition
        el_amounts = sorted(reduced.values())
        if len(el_amounts) < 3:
            return False
        # ABO3: one A(×1), one B(×1), one X(×3) per formula unit
        return abs(el_amounts[-1] - 3.0) < 0.5 and abs(el_amounts[-2] - 1.0) < 0.3
    except Exception:
        return False
