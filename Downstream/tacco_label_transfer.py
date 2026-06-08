#!/usr/bin/env python3
"""
TACCO-based label transfer from atlas to one query AnnData file.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import anndata as ad
import pandas as pd
import scanpy as sc
import tacco as tc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TACCO label transfer on one query sample.")
    parser.add_argument("--atlas_path", required=True, help="Path to atlas .h5ad file.")
    parser.add_argument("--query_path", required=True, help="Path to query .h5ad file.")
    parser.add_argument("--out_dir", required=True, help="Root output directory.")
    parser.add_argument("--sample_name", default=None, help="Optional sample name override.")
    parser.add_argument(
        "--label_keys",
        nargs="+",
        required=True,
        help="Atlas obs label columns to transfer, e.g. Level_1 Level_2 Level_3 Level_4.",
    )
    parser.add_argument(
        "--method",
        default="OT",
        help="TACCO annotation method (default: OT).",
    )
    parser.add_argument(
        "--counts_layer",
        default="counts",
        help="Preferred counts layer. Falls back to X if absent.",
    )
    parser.add_argument(
        "--assume_valid_counts",
        action="store_true",
        help="Skip TACCO integer-count validation (use with trusted counts layers).",
    )
    return parser.parse_args()


def _log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} | {msg}", flush=True)


def _ensure_counts(adata: ad.AnnData, preferred_layer: str) -> None:
    if preferred_layer in adata.layers:
        adata.layers["counts"] = adata.layers[preferred_layer].copy()
    elif "counts" in adata.layers:
        pass
    else:
        adata.layers["counts"] = adata.X.copy()


def _prep_for_tacco(adata: ad.AnnData) -> ad.AnnData:
    work = adata.copy()
    if "counts" in work.layers:
        # TACCO expects raw count-like input for its own preprocessing path.
        work.X = work.layers["counts"].copy()
    return work


def main() -> None:
    args = parse_args()
    sample_name = args.sample_name or os.path.basename(args.query_path).replace(".h5ad", "")

    h5ad_dir = os.path.join(args.out_dir, "h5ad")
    csv_dir = os.path.join(args.out_dir, "csv")
    status_dir = os.path.join(args.out_dir, "status")
    logs_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(h5ad_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(status_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    _log(f"Loading atlas: {args.atlas_path}")
    atlas = sc.read_h5ad(args.atlas_path)
    _log(f"Loading query: {args.query_path}")
    query = sc.read_h5ad(args.query_path)

    _ensure_counts(atlas, args.counts_layer)
    _ensure_counts(query, args.counts_layer)

    shared = atlas.var_names.intersection(query.var_names)
    if len(shared) == 0:
        raise ValueError("No shared genes between atlas and query.")
    atlas = atlas[:, shared].copy()
    query = query[:, shared].copy()

    for key in args.label_keys:
        if key not in atlas.obs:
            raise KeyError(f"Label key '{key}' not found in atlas.obs")
        atlas.obs[key] = atlas.obs[key].astype(str)

    atlas_tacco = _prep_for_tacco(atlas)
    query_tacco = _prep_for_tacco(query)

    result_df = pd.DataFrame(index=query.obs_names.copy())
    result_df.index.name = "cell_id"

    for key in args.label_keys:
        score_key = f"tacco_scores_{key}"
        pred_key = f"predicted_{key}"
        conf_key = f"confidence_{key}"
        _log(f"Running TACCO for {key} with method={args.method}")
        tc.tl.annotate(
            query_tacco,
            atlas_tacco,
            annotation_key=key,
            result_key=score_key,
            method=args.method,
            assume_valid_counts=args.assume_valid_counts,
        )
        scores = query_tacco.obsm[score_key].copy()
        if not isinstance(scores, pd.DataFrame):
            scores = pd.DataFrame(scores, index=query_tacco.obs_names)
        pred = scores.idxmax(axis=1).astype(str)
        conf = scores.max(axis=1).astype(float)

        query.obs[pred_key] = pred.reindex(query.obs_names).values
        query.obs[conf_key] = conf.reindex(query.obs_names).values
        query.obsm[score_key] = scores.reindex(query.obs_names)

        result_df[pred_key] = query.obs[pred_key]
        result_df[conf_key] = query.obs[conf_key]

        score_out = os.path.join(csv_dir, f"{sample_name}_{key}_scores.csv")
        query.obsm[score_key].to_csv(score_out)
        _log(f"Saved score matrix: {score_out}")

    pred_out = os.path.join(csv_dir, f"{sample_name}.csv")
    result_df.to_csv(pred_out)
    _log(f"Saved summary CSV: {pred_out}")

    out_h5ad = os.path.join(h5ad_dir, f"{sample_name}.h5ad")
    query.write_h5ad(out_h5ad)
    _log(f"Saved annotated query: {out_h5ad}")

    with open(os.path.join(status_dir, f"{sample_name}.done"), "w", encoding="utf-8") as handle:
        handle.write(f"status\tcompleted\ncompleted_at\t{datetime.now(UTC).isoformat()}\n")
    _log("Done.")


if __name__ == "__main__":
    main()
