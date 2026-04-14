#!/usr/bin/env python3
"""Prepare benchmark assets required for RL training.

Builds the three asset files that can be created without an external API key:
  - benchmarks/assets/mp_20_all_structure.json.gz   (reference structures for novelty check)
  - benchmarks/assets/mp_20_all_structure_features.pt  (VAE latent features)
  - benchmarks/assets/mp_20_all_composition_features.pt

These are generated from the CSV files already in data/mp-20/.

The phase diagram needed by EnergyReward is NOT built here — it requires the
full Materials Project database. See the note at the end of this script.

Usage:
    uv run python scripts/prepare_assets.py
    uv run python scripts/prepare_assets.py --batch-size 256  # lower if OOM
"""

import argparse
import sys
from pathlib import Path

import torch
from monty.serialization import dumpfn, loadfn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "benchmarks" / "assets"
DATA_DIR = REPO_ROOT / "data" / "mp-20"

REFERENCE_STRUCTURES_PATH = ASSETS_DIR / "mp_20_all_structure.json.gz"
STRUCTURE_FEATURES_PATH = ASSETS_DIR / "mp_20_all_structure_features.pt"
COMPOSITION_FEATURES_PATH = ASSETS_DIR / "mp_20_all_composition_features.pt"


# ---------------------------------------------------------------------------
# Step 1: reference structures
# ---------------------------------------------------------------------------

def build_reference_structures() -> None:
    """Parse CIF strings from train/val/test CSVs and save as a JSON.gz."""
    if REFERENCE_STRUCTURES_PATH.exists():
        print(f"  Already exists: {REFERENCE_STRUCTURES_PATH.name}")
        return

    import pandas as pd
    from pymatgen.core import Structure

    structures = []
    for split in ("train", "val", "test"):
        csv_path = DATA_DIR / f"{split}.csv"
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        print(f"  Parsing {len(df)} structures from {split}.csv ...")
        for cif in tqdm(df["cif"], desc=f"  {split}", leave=False):
            try:
                structures.append(Structure.from_str(cif, fmt="cif"))
            except Exception:
                pass  # skip malformed CIFs

    if not structures:
        print("  ERROR: no structures parsed. Check that data/mp-20/*.csv exist.")
        sys.exit(1)

    print(f"  Saving {len(structures)} structures → {REFERENCE_STRUCTURES_PATH.name}")
    dumpfn(structures, REFERENCE_STRUCTURES_PATH)
    print(f"  Done.")


# ---------------------------------------------------------------------------
# Step 2: VAE features
# ---------------------------------------------------------------------------

def build_features(batch_size: int) -> None:
    """Encode reference structures with the pre-trained VAE and save features."""
    if STRUCTURE_FEATURES_PATH.exists() and COMPOSITION_FEATURES_PATH.exists():
        print(f"  Already exist: {STRUCTURE_FEATURES_PATH.name}, {COMPOSITION_FEATURES_PATH.name}")
        return

    # Import here so errors surface cleanly
    from src.utils.featurizer import featurize

    print(f"  Loading structures from {REFERENCE_STRUCTURES_PATH.name} ...")
    structures = loadfn(REFERENCE_STRUCTURES_PATH)

    print(f"  Encoding {len(structures)} structures with the pre-trained VAE ...")
    print(f"  (batch_size={batch_size}; reduce with --batch-size if you run out of memory)")
    features = featurize(structures, batch_size=batch_size)

    torch.save(features["structure_features"], STRUCTURE_FEATURES_PATH)
    print(f"  Saved: {STRUCTURE_FEATURES_PATH.name}")

    torch.save(features["composition_features"], COMPOSITION_FEATURES_PATH)
    print(f"  Saved: {COMPOSITION_FEATURES_PATH.name}")


# ---------------------------------------------------------------------------
# Step 3: phase diagram (informational only)
# ---------------------------------------------------------------------------

PHASE_DIAGRAM_PATH = ASSETS_DIR / "ppd-mp_all_entries_uncorrected_250409.pkl.gz"

PHASE_DIAGRAM_INSTRUCTIONS = """
  ── Phase diagram (for EnergyReward) ─────────────────────────────────────
  The phase diagram file is NOT built by this script — it requires downloading
  the full Materials Project database (~200k entries), which takes ~1 hour.

  To build it (requires an MP_API_KEY):

      export MP_API_KEY=<your-key>
      uv run python benchmarks/build_phase_diagram.py

  Until the file exists, EnergyReward is excluded from the training configs.
  You can re-add it later by setting:
      - _target_: src.rl_module.components.EnergyReward
        weight: 1.0
        normalize_fn: norm
  in configs/experiment/mp_20/rl_ferroelectric.yaml after building the file.
  ──────────────────────────────────────────────────────────────────────────
"""


def check_phase_diagram() -> None:
    if PHASE_DIAGRAM_PATH.exists():
        print(f"  Found: {PHASE_DIAGRAM_PATH.name}")
    else:
        print(PHASE_DIAGRAM_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="VAE encoding batch size (default: 512; lower if OOM)")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Assets directory: {ASSETS_DIR}\n")

    print("[1/3] Building reference structures ...")
    build_reference_structures()

    print("\n[2/3] Building VAE features ...")
    build_features(batch_size=args.batch_size)

    print("\n[3/3] Phase diagram check ...")
    check_phase_diagram()

    print("\nAsset preparation complete.")
    print("Start training with:")
    print("  ./scripts/train_rl_ferroelectric.sh")


if __name__ == "__main__":
    main()
