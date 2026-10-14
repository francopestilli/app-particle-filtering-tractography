#!/usr/bin/env python3
"""
run_pft.py — Particle Filtering Tractography using dipy directly.

Parallelism strategy (fork-based):
  - fODF is loaded ONCE in the main process, direction getter and stopping
    criterion are built ONCE, then N workers are forked.  Workers inherit
    all data via COW — no redundant I/O or import overhead.
  - Workers receive only their seed chunk (small numpy array) via IPC.

5TT volume convention (vol index):
    0 = cortical GM    → include map
    1 = subcortical GM → include map
    2 = WM/brainstem   → seed mask
    3 = CSF            → exclude map
    4 = pathological   → ignored
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

# Module-level dict populated in main() BEFORE forking workers.
# Forked children inherit it via COW — nothing is pickled.
_SHARED = {}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="PFT tractography via dipy (fork-parallel)")
    ap.add_argument("--fodf",         required=True,  help="fODF / CSD SH volume (.nii.gz)")
    ap.add_argument("--mask_5tt",     required=True,  help="5TT probability map (.nii.gz, 4-vol)")
    ap.add_argument("--seed_mask",    default=None,   help="Seeding mask override (.nii.gz)")
    ap.add_argument("--output_dir",   default=".",    help="Directory for tractogram/ output")
    ap.add_argument("--config",       default=None,   help="config.json (overrides all other flags)")
    ap.add_argument("--jobs",         type=int, default=None,
                    help="CPU cores (default: all available)")
    ap.add_argument("--algo",           default="prob",  choices=["prob", "det"])
    ap.add_argument("--seeding_type",   default="npv",   choices=["npv", "nt"])
    ap.add_argument("--nbr_seeds",      type=int,   default=50)
    ap.add_argument("--step",           type=float, default=0.2)
    ap.add_argument("--theta",          type=float, default=20.0)
    ap.add_argument("--sfthres",        type=float, default=0.1)
    ap.add_argument("--sfthres_init",   type=float, default=0.5)
    ap.add_argument("--min_length",     type=float, default=20.0)
    ap.add_argument("--max_length",     type=float, default=200.0)
    ap.add_argument("--particles",      type=int,   default=15)
    ap.add_argument("--back_tracking",  type=float, default=2.0)
    ap.add_argument("--forward_tracking", type=float, default=1.0)
    ap.add_argument("--compress",       type=bool,  default=True)
    ap.add_argument("--compress_value", type=float, default=0.2)
    ap.add_argument("--random_seed",    type=int,   default=0)
    ap.add_argument("--sh_basis",       default="descoteaux07",
                    choices=["descoteaux07", "tournier07"])
    ap.add_argument("--dry_run",        action="store_true")
    return ap.parse_args()


def merge_config(args):
    if args.config is None:
        return
    cfg = json.loads(Path(args.config).read_text())
    for key, val in cfg.items():
        if hasattr(args, key) and val not in (None, ""):
            setattr(args, key, val)


# ---------------------------------------------------------------------------
# Map derivation
# ---------------------------------------------------------------------------

def derive_maps(mask_5tt_data):
    gm  = mask_5tt_data[..., 0] + mask_5tt_data[..., 1]
    inc = np.clip(gm, 0.0, 1.0).astype(np.float64)
    exc = np.squeeze(mask_5tt_data[..., 3]).astype(np.float64)
    wm  = (mask_5tt_data[..., 2] > 0).astype(np.uint8)
    # GM-WM interface: voxels with both WM and GM probability > 0
    interface = ((mask_5tt_data[..., 2] > 0) & (gm > 0)).astype(np.uint8)
    return inc, exc, wm, interface


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def make_seeds(seed_vol, affine, seeding_type, nbr_seeds, rng=0):
    from dipy.tracking.utils import random_seeds_from_mask
    # dipy 1.12+: seed_count_per_voxel=True means seeds_count is per voxel (npv);
    # False means seeds_count is the total (nt).
    return random_seeds_from_mask(
        seed_vol, affine=affine,
        seeds_count=nbr_seeds,
        seed_count_per_voxel=(seeding_type == "npv"),
        random_seed=rng,
    )


# ---------------------------------------------------------------------------
# Worker — runs in forked child; _SHARED is already populated via COW
# ---------------------------------------------------------------------------

def _track_batch(chunk):
    from dipy.tracking.local_tracking import ParticleFilteringTracking
    p   = _SHARED["params"]
    dg  = _SHARED["dg"]
    sc  = _SHARED["sc"]
    aff = _SHARED["affine"]

    tracker = ParticleFilteringTracking(
        direction_getter=dg,
        stopping_criterion=sc,
        seeds=chunk,
        affine=aff,
        step_size=p["step"],
        maxlen=int(p["max_length"] / p["step"]) + 1,
        particle_count=p["particles"],
        pft_back_tracking_dist=p["back_tracking"],
        pft_front_tracking_dist=p["forward_tracking"],
        return_all=False,
        random_seed=p["random_seed"],
    )
    min_pts = int(p["min_length"] / p["step"])
    return [s for s in tracker if len(s) >= min_pts]


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress_streamlines(streamlines, tol=0.2):
    try:
        from dipy.tracking.streamlinespeed import compress_streamlines as _c
        return [_c(s, tol) for s in streamlines]
    except ImportError:
        return streamlines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    merge_config(args)

    n_jobs  = args.jobs or mp.cpu_count()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tract_dir = out_dir / "tractogram"
    tract_dir.mkdir(exist_ok=True)

    print(f"=== PFT Tractography (dipy {__import__('dipy').__version__}) ===")
    print(f"  fodf       : {args.fodf}")
    print(f"  mask_5tt   : {args.mask_5tt}")
    print(f"  algo       : {args.algo}")
    seed_src = args.seed_mask if args.seed_mask else "GM-WM interface"
    print(f"  seeds/vox  : {args.nbr_seeds}  type={args.seeding_type}  src={seed_src}")
    print(f"  step       : {args.step} mm   theta={args.theta}°")
    print(f"  length     : [{args.min_length}, {args.max_length}] mm")
    print(f"  particles  : {args.particles}")
    print(f"  jobs       : {n_jobs}")
    print(f"  output     : {out_dir}")
    if args.dry_run:
        print("\n(dry run — exiting)"); return

    # ------------------------------------------------------------------
    # [1] Load volumes in the main process — done ONCE
    # ------------------------------------------------------------------
    print("\n[1/5] Loading fODF ...", flush=True)
    fodf_img = nib.load(args.fodf)
    sh_coeff = fodf_img.get_fdata(dtype=np.float32)
    affine   = fodf_img.affine
    print(f"      shape={sh_coeff.shape}", flush=True)

    print("[2/5] Loading 5TT ...", flush=True)
    tt5_data = nib.load(args.mask_5tt).get_fdata(dtype=np.float32)
    inc_map, exc_map, wm_bin, interface_bin = derive_maps(tt5_data)

    # ------------------------------------------------------------------
    # [2] Build direction getter and stopping criterion ONCE
    # ------------------------------------------------------------------
    print("[3/5] Building direction getter ...", flush=True)
    from dipy.direction import (ProbabilisticDirectionGetter,
                                DeterministicMaximumDirectionGetter)
    from dipy.data import get_sphere
    from dipy.tracking.stopping_criterion import ActStoppingCriterion

    sphere = get_sphere("repulsion724")

    if args.algo == "prob":
        dg = ProbabilisticDirectionGetter.from_shcoeff(
            sh_coeff, max_angle=args.theta, sphere=sphere,
            basis_type=args.sh_basis, pmf_threshold=args.sfthres,
            legacy=False,
        )
    else:
        dg = DeterministicMaximumDirectionGetter.from_shcoeff(
            sh_coeff, max_angle=args.theta, sphere=sphere,
            basis_type=args.sh_basis, pmf_threshold=args.sfthres,
            legacy=False,
        )

    if args.seed_mask:
        seed_vol = (nib.load(args.seed_mask).get_fdata(dtype=np.float32) > 0).astype(np.uint8)
    else:
        seed_vol = interface_bin  # GM-WM interface seeding

    sc = ActStoppingCriterion(inc_map, exc_map)

    # ------------------------------------------------------------------
    # [3] Generate seeds
    # ------------------------------------------------------------------
    print("[4/5] Generating seeds ...", flush=True)
    seeds = make_seeds(seed_vol, affine, args.seeding_type, args.nbr_seeds,
                       rng=args.random_seed)
    print(f"      {len(seeds):,} seeds total", flush=True)

    # ------------------------------------------------------------------
    # Populate shared state BEFORE forking — workers inherit via COW
    # ------------------------------------------------------------------
    params = dict(
        algo=args.algo, theta=args.theta, sfthres=args.sfthres,
        sfthres_init=args.sfthres_init, step=args.step,
        min_length=args.min_length, max_length=args.max_length,
        particles=args.particles, back_tracking=args.back_tracking,
        forward_tracking=args.forward_tracking,
        random_seed=args.random_seed, sh_basis=args.sh_basis,
    )
    _SHARED.update(dg=dg, sc=sc, affine=affine, params=params)

    # Limit numpy threads in workers so 128 workers don't fight each other
    os.environ["OMP_NUM_THREADS"]     = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"

    # ------------------------------------------------------------------
    # [4] Fork workers — each inherits dg, sc, affine, params (COW)
    # ------------------------------------------------------------------
    print(f"[5/5] Tracking with {n_jobs} workers (fork) ...", flush=True)
    chunks = np.array_split(seeds, n_jobs)
    chunks = [c for c in chunks if len(c) > 0]

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(n_jobs) as pool:
        results = pool.map(_track_batch, chunks)

    all_streamlines = [s for batch in results for s in batch]
    elapsed = time.time() - t0
    print(f"      {len(all_streamlines):,} streamlines in {elapsed:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Compress and save
    # ------------------------------------------------------------------
    if args.compress:
        print(f"Compressing (tol={args.compress_value} mm) ...", flush=True)
        all_streamlines = compress_streamlines(all_streamlines, args.compress_value)

    out_tck = tract_dir / "track.tck"
    print(f"Saving → {out_tck}", flush=True)
    from dipy.io.streamline import save_tractogram
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    sft = StatefulTractogram(all_streamlines, fodf_img, Space.RASMM)
    save_tractogram(sft, str(out_tck), bbox_valid_check=False)
    print(f"\nDone. Tractogram: {out_tck}  ({len(all_streamlines):,} streamlines)", flush=True)


if __name__ == "__main__":
    main()
