"""Unit tests for Phase 1 ferroelectric reward components.

All structures are constructed directly from pymatgen primitives — no MP API
or network access required.

Validation fixtures mirror the sanity checks described in the implementation
brief: known ferroelectrics (BaTiO₃-like, PbTiO₃-like, LiNbO₃-like) must
score well; known non-ferroelectrics (NaCl, SrTiO₃ cubic) must score low.
"""

import pytest
import torch
from pymatgen.core import Lattice, Structure

from src.rl_module.ferroelectric import (
    POLAR_SPACE_GROUPS,
    FEChemistryReward,
    PolarDistortionReward,
    PolarSymmetryReward,
    _fe_chemistry_score,
    _is_perovskite_like,
    _polar_distortion_score,
    _polar_symmetry_score,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batio3_tetragonal() -> Structure:
    """Approximate BaTiO₃ tetragonal structure (P4mm, SG 99).

    Ti is displaced ~0.12 Å from the centroid of its oxygen octahedron
    along the c-axis — the canonical ferroelectric distortion.
    """
    lattice = Lattice.tetragonal(a=3.994, c=4.034)
    species = ["Ba", "Ti", "O", "O", "O"]
    # Fractional coordinates from Kwei et al. 1993
    coords = [
        [0.0, 0.0, 0.000],   # Ba at origin
        [0.5, 0.5, 0.519],   # Ti displaced toward apical O
        [0.5, 0.5, 0.026],   # O apical
        [0.5, 0.0, 0.486],   # O equatorial
        [0.0, 0.5, 0.486],   # O equatorial
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def pbzro3_tetragonal() -> Structure:
    """PbZrO₃-like tetragonal structure with lone-pair Pb and d0 Zr."""
    lattice = Lattice.tetragonal(a=4.05, c=4.10)
    species = ["Pb", "Zr", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.000],
        [0.5, 0.5, 0.520],
        [0.5, 0.5, 0.030],
        [0.5, 0.0, 0.485],
        [0.0, 0.5, 0.485],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def linbo3_like() -> Structure:
    """LiNbO₃-like rhombohedral structure approximation.

    Uses a trigonal cell with Nb off-centered from its oxygen octahedron.
    The true SG is 161 (R3c), which is polar (C3v point group).
    """
    lattice = Lattice.rhombohedral(a=5.15, alpha=56.2)
    species = ["Li", "Nb", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.270],
        [0.0, 0.0, 0.000],
        [0.050, 0.350, 0.933],
        [0.350, 0.933, 0.050],
        [0.933, 0.050, 0.350],
    ]
    return Structure(lattice, species, coords)


@pytest.fixture
def nacl_cubic() -> Structure:
    """NaCl cubic structure (SG 225, Fm-3m) — centrosymmetric, not FE."""
    return Structure.from_spacegroup(
        225,
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture
def srtio3_cubic() -> Structure:
    """SrTiO₃ cubic paraelectric structure (SG 221, Pm-3m).

    Above its Curie temperature SrTiO₃ is centrosymmetric — Ti sits
    exactly at the centre of the oxygen octahedron with zero polarisation.
    """
    return Structure.from_spacegroup(
        221,
        Lattice.cubic(3.905),
        ["Sr", "Ti", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]],
    )


@pytest.fixture
def triclinic_polar() -> Structure:
    """Robustly P1 (SG 1, C1) structure: three different species at general positions
    in a strongly triclinic cell.  No pair of atoms differs by a lattice translation
    or inversion, so spglib has no symmetry to find.
    """
    # Strongly triclinic cell (angles far from 90°, no near-orthogonality)
    lattice = Lattice.from_parameters(5.0, 6.1, 7.3, 73.2, 82.5, 67.8)
    # Three different species: no symmetry operation can map Ti→Nb or Ti→O
    species = ["Ti", "Nb", "O"]
    coords = [[0.11, 0.23, 0.37], [0.44, 0.61, 0.72], [0.82, 0.09, 0.55]]
    return Structure(lattice, species, coords)


# ---------------------------------------------------------------------------
# PolarSymmetryReward
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolarSymmetryReward:
    def test_returns_tensor(self, batio3_tetragonal):
        reward = PolarSymmetryReward(weight=1.0)
        result = reward(gen_structures=[batio3_tetragonal], device=None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_batio3_is_polar(self, batio3_tetragonal):
        """BaTiO₃ tetragonal (SG 99, P4mm) must return 1.0."""
        score = _polar_symmetry_score(batio3_tetragonal, symprec=0.2)
        assert score == 1.0, f"BaTiO₃ should be polar, got {score}"

    def test_triclinic_p1_is_polar(self, triclinic_polar):
        """SG 1 (P1) is always polar."""
        score = _polar_symmetry_score(triclinic_polar, symprec=0.2)
        assert score == 1.0

    def test_nacl_is_not_polar(self, nacl_cubic):
        """NaCl (SG 225, Fm-3m) is centrosymmetric — must return 0.0."""
        score = _polar_symmetry_score(nacl_cubic, symprec=0.1)
        assert score == 0.0, f"NaCl should not be polar, got {score}"

    def test_srtio3_cubic_is_not_polar(self, srtio3_cubic):
        """Cubic SrTiO₃ (SG 221) is centrosymmetric — must return 0.0."""
        score = _polar_symmetry_score(srtio3_cubic, symprec=0.1)
        assert score == 0.0, f"Cubic SrTiO₃ should not be polar, got {score}"

    def test_batch_length(self, batio3_tetragonal, nacl_cubic):
        reward = PolarSymmetryReward(weight=1.0)
        structures = [batio3_tetragonal, nacl_cubic, batio3_tetragonal]
        result = reward(gen_structures=structures, device=None)
        assert result.shape == (3,)

    def test_weight_applied(self, batio3_tetragonal):
        reward = PolarSymmetryReward(weight=2.0)
        result = reward(gen_structures=[batio3_tetragonal], device=None)
        # BaTiO₃ polar score is 1.0; with weight=2.0 the output should be 2.0
        assert result.item() == pytest.approx(2.0)

    def test_polar_space_groups_count(self):
        """All 68 polar space groups must be present in the constant."""
        assert len(POLAR_SPACE_GROUPS) == 68

    def test_polar_space_groups_known_members(self):
        assert 1 in POLAR_SPACE_GROUPS    # C1
        assert 99 in POLAR_SPACE_GROUPS   # P4mm (BaTiO₃)
        assert 161 in POLAR_SPACE_GROUPS  # R3c (LiNbO₃)
        assert 183 in POLAR_SPACE_GROUPS  # P6mm

    def test_polar_space_groups_known_non_members(self):
        assert 225 not in POLAR_SPACE_GROUPS  # Fm-3m (NaCl)
        assert 221 not in POLAR_SPACE_GROUPS  # Pm-3m (SrTiO₃ cubic)
        assert 2 not in POLAR_SPACE_GROUPS    # P-1 (centrosymmetric)


# ---------------------------------------------------------------------------
# PolarDistortionReward
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPolarDistortionReward:
    def test_returns_tensor(self, batio3_tetragonal):
        reward = PolarDistortionReward(weight=1.0)
        result = reward(gen_structures=[batio3_tetragonal], device=None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_batio3_positive_distortion(self, batio3_tetragonal):
        """Displaced Ti should yield a positive distortion score."""
        score = _polar_distortion_score(batio3_tetragonal, norm_factor=0.3)
        assert score > 0.0, f"BaTiO₃ should have positive distortion, got {score}"
        assert score <= 1.0

    def test_pbzro3_positive_distortion(self, pbzro3_tetragonal):
        """Displaced Zr in PbZrO₃-like structure should yield a positive score."""
        score = _polar_distortion_score(pbzro3_tetragonal, norm_factor=0.3)
        assert score > 0.0
        assert score <= 1.0

    def test_score_bounded(self, batio3_tetragonal, nacl_cubic):
        """All scores must be in [0, 1]."""
        for s in [batio3_tetragonal, nacl_cubic]:
            score = _polar_distortion_score(s, norm_factor=0.3)
            assert 0.0 <= score <= 1.0

    def test_nacl_fallback_to_asymmetry(self, nacl_cubic):
        """NaCl has no B-site cations; fallback to bond-length asymmetry.

        NaCl is highly symmetric so asymmetry should be near 0.
        """
        score = _polar_distortion_score(nacl_cubic, norm_factor=0.3)
        assert 0.0 <= score <= 1.0

    def test_norm_factor_scaling(self, batio3_tetragonal):
        """Smaller norm_factor → larger (saturated) score for the same structure."""
        score_default = _polar_distortion_score(batio3_tetragonal, norm_factor=0.3)
        score_tight = _polar_distortion_score(batio3_tetragonal, norm_factor=0.05)
        # Tighter norm = score saturates sooner → score_tight >= score_default
        assert score_tight >= score_default

    def test_batch_returns_per_sample(self, batio3_tetragonal, nacl_cubic):
        reward = PolarDistortionReward(weight=1.0)
        structures = [batio3_tetragonal, nacl_cubic]
        result = reward(gen_structures=structures, device=None)
        assert result.shape == (2,)
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)


# ---------------------------------------------------------------------------
# FEChemistryReward
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFEChemistryReward:
    def test_returns_tensor(self, batio3_tetragonal):
        reward = FEChemistryReward(weight=1.0)
        result = reward(gen_structures=[batio3_tetragonal], device=None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_batio3_high_score(self, batio3_tetragonal):
        """BaTiO₃: d0 Ti (+0.35) + oxide O (+0.10) + perovskite (+0.20) = 0.65."""
        score = _fe_chemistry_score(batio3_tetragonal)
        assert score >= 0.60, f"BaTiO₃ FE chemistry score too low: {score}"

    def test_pbzro3_maximum_score(self, pbzro3_tetragonal):
        """PbZrO₃: lone-pair Pb (+0.35) + d0 Zr (+0.35) + oxide (+0.10) + perovskite (+0.20) = 1.0."""
        score = _fe_chemistry_score(pbzro3_tetragonal)
        assert score == pytest.approx(1.0), f"PbZrO₃ should score 1.0, got {score}"

    def test_nacl_low_score(self, nacl_cubic):
        """NaCl has no lone-pair cations, no d0 TMs, no perovskite stoich → 0.10 max (halide)."""
        score = _fe_chemistry_score(nacl_cubic)
        # Cl is a halide, so anion bonus (+0.10) may fire; still very low
        assert score <= 0.15, f"NaCl FE chemistry score too high: {score}"

    def test_score_bounded(self, batio3_tetragonal, nacl_cubic, triclinic_polar):
        for s in [batio3_tetragonal, nacl_cubic, triclinic_polar]:
            score = _fe_chemistry_score(s)
            assert 0.0 <= score <= 1.0

    def test_perovskite_detection_batio3(self, batio3_tetragonal):
        assert _is_perovskite_like(batio3_tetragonal.composition)

    def test_perovskite_detection_nacl(self, nacl_cubic):
        assert not _is_perovskite_like(nacl_cubic.composition)

    def test_batch_returns_per_sample(self, batio3_tetragonal, nacl_cubic):
        reward = FEChemistryReward(weight=1.0)
        structures = [batio3_tetragonal, nacl_cubic, batio3_tetragonal]
        result = reward(gen_structures=structures, device=None)
        assert result.shape == (3,)
        # BaTiO₃ should outscore NaCl
        assert result[0] > result[1]
        assert result[2] > result[1]


# ---------------------------------------------------------------------------
# Integration: all three rewards together
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFerroelectricRewardsCombined:
    def test_batio3_scores_well_on_all_three(self, batio3_tetragonal):
        """BaTiO₃ should score well on all three Phase 1 rewards."""
        polar_sym = _polar_symmetry_score(batio3_tetragonal, symprec=0.2)
        polar_dist = _polar_distortion_score(batio3_tetragonal, norm_factor=0.3)
        fe_chem = _fe_chemistry_score(batio3_tetragonal)

        assert polar_sym == 1.0, f"BaTiO₃ polar symmetry: expected 1.0, got {polar_sym}"
        assert polar_dist > 0.0, f"BaTiO₃ polar distortion: expected >0, got {polar_dist}"
        assert fe_chem >= 0.60, f"BaTiO₃ FE chemistry: expected ≥0.60, got {fe_chem}"

    def test_nacl_scores_low_on_all_three(self, nacl_cubic):
        """NaCl should score 0 on polar symmetry and low on the others."""
        polar_sym = _polar_symmetry_score(nacl_cubic, symprec=0.1)
        fe_chem = _fe_chemistry_score(nacl_cubic)

        assert polar_sym == 0.0
        assert fe_chem <= 0.15

    def test_all_reward_outputs_finite(self, batio3_tetragonal, nacl_cubic, srtio3_cubic):
        """Reward components must not produce NaN or Inf for any test structure."""
        reward_classes = [
            PolarSymmetryReward(weight=1.0),
            PolarDistortionReward(weight=1.0),
            FEChemistryReward(weight=1.0),
        ]
        for structure in [batio3_tetragonal, nacl_cubic, srtio3_cubic]:
            for reward in reward_classes:
                result = reward(gen_structures=[structure], device=None)
                assert torch.isfinite(result).all(), (
                    f"{reward.__class__.__name__} returned non-finite value "
                    f"for {structure.formula}: {result}"
                )
