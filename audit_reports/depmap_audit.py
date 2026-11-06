#!/usr/bin/env python3
"""Audit a DepMap model-by-gene matrix without loading it all into memory.

Requires: numpy, pandas. Run --help for options.
Outputs are descriptive summaries, NOT cross-validation training baselines.
NaN targets remain missing; they are never replaced by zero.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def audit(matrix, model_file, out, chunksize=100, target="dependency", release="26Q3"):
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    matrix, model_file, out = Path(matrix), Path(model_file), Path(out)
    # Read the ORIGINAL CSV header, avoiding spreadsheet column limits and
    # pandas' automatic renaming of duplicate column names.
    with matrix.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) < 2:
        raise ValueError("Expected ModelID followed by gene columns")
    genes = header[1:]
    if any(not name.strip() for name in genes):
        raise ValueError("Empty gene column label")
    duplicates = [g for g, n in Counter(genes).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate gene column labels: {duplicates[:5]}")

    # A small metadata table is loaded once. It is not the target matrix.
    metadata = pd.read_csv(model_file, dtype="string", encoding="utf-8-sig")
    required = {"ModelID", "OncotreeLineage"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Model.csv must contain {sorted(required)}")
    metadata["ModelID"] = metadata["ModelID"].str.strip()
    if (metadata.ModelID.isna() | metadata.ModelID.eq("")).any():
        raise ValueError("Missing ModelID in Model.csv")
    if metadata.ModelID.duplicated().any():
        raise ValueError("Duplicate ModelID in Model.csv; resolve before joining")
    metadata["OncotreeLineage"] = metadata.OncotreeLineage.str.strip().replace("", pd.NA)
    metadata = metadata.set_index("ModelID")

    p = len(genes)
    count = np.zeros(p, dtype=np.int64)
    mean = np.zeros(p, dtype=np.float64)
    m2 = np.zeros(p, dtype=np.float64)
    minimum = np.full(p, np.inf)
    maximum = np.full(p, -np.inf)
    seen = set()
    row_tables = []
    previews = []
    n = outside_range = 0
    # Name the first column explicitly: DepMap exports may have a blank header.
    names = ["ModelID"] + genes
    types = {gene: "float64" for gene in genes}
    types["ModelID"] = "string"
    with pd.read_csv(matrix, header=0, names=names, dtype=types,
                     chunksize=chunksize, encoding="utf-8-sig") as reader:
        for chunk in reader:
            ids = chunk.pop("ModelID").str.strip()
            if (ids.isna() | ids.eq("")).any():
                raise ValueError("Missing ModelID in dependency matrix")
            if ids.duplicated().any() or seen.intersection(ids.tolist()):
                raise ValueError("Duplicate ModelID in dependency matrix")
            if not ids.str.startswith("ACH-").all():
                raise ValueError("Expected ACH- ModelIDs in first column; check input format")
            seen.update(ids.tolist())
            values = chunk.to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError("Infinite score encountered; inspect source data")
            valid = ~np.isnan(values)
            batch_count = valid.sum(axis=0)
            batch_sum = np.where(valid, values, 0).sum(axis=0)
            batch_mean = np.divide(batch_sum, batch_count,
                                   out=np.zeros(p), where=batch_count > 0)
            centered = np.where(valid, values - batch_mean, 0)
            batch_m2 = np.square(centered).sum(axis=0)
            # Merge batch moments, weighted by NON-MISSING observations per gene.
            # Averaging batch means directly would be wrong for unequal batches.
            total = count + batch_count
            delta = batch_mean - mean
            mean += delta * np.divide(batch_count, total,
                                     out=np.zeros(p), where=total > 0)
            m2 += batch_m2 + delta**2 * np.divide(
                count * batch_count, total, out=np.zeros(p), where=total > 0)
            count = total
            minimum = np.minimum(minimum, np.where(valid, values, np.inf).min(axis=0))
            maximum = np.maximum(maximum, np.where(valid, values, -np.inf).max(axis=0))
            outside_range += int(((values < 0) | (values > 1)).sum())
            observed = valid.sum(axis=1)
            row_tables.append(pd.DataFrame({
                "ModelID": ids.to_numpy(), "n_observed_genes": observed,
                "n_missing_genes": p - observed,
                "pct_observed_genes": 100 * observed / p,
            }))
            if n < 5:
                preview = chunk.iloc[:5 - n, :8].copy()
                preview.insert(0, "ModelID", ids.iloc[:len(preview)].to_numpy())
                previews.append(preview)
            n += len(chunk)
            print(f"Processed {n:,} models; {p:,} gene columns", flush=True)
    if n == 0:
        raise ValueError("No model rows found")

    rows = pd.concat(row_tables, ignore_index=True)
    # Reindex metadata to EXACTLY the matrix cohort, preserving its row order.
    selected = metadata.reindex(rows.ModelID).reset_index()
    selected["metadata_found"] = selected.ModelID.isin(metadata.index)
    rows = rows.merge(selected, on="ModelID", how="left", validate="one_to_one", sort=False)
    lineage_counts = (rows.groupby("OncotreeLineage", dropna=False)
                      .size().rename("n_models").reset_index()
                      .sort_values("n_models", ascending=False))
    lineage_counts["pct_of_matrix_models"] = 100 * lineage_counts.n_models / n
    std = np.sqrt(np.divide(m2, count, out=np.full(p, np.nan), where=count > 0))
    stats = pd.DataFrame({
        "gene_column": genes,
        "n_observed_models": count, "n_missing_models": n - count,
        "pct_observed_models": 100 * count / n,
        "mean": np.where(count > 0, mean, np.nan), "std_population": std,
        "min": np.where(count > 0, minimum, np.nan),
        "max": np.where(count > 0, maximum, np.nan),
    })
    # Preserve full gene identifiers; do not collapse matching symbols.
    parsed = stats.gene_column.str.extract(r"^(.*) \((\d+)\)$")
    stats.insert(1, "gene_symbol_if_parsed", parsed[0])
    stats.insert(2, "entrez_id_if_parsed", parsed[1])
    summary = {
        "release_label_user_supplied": release,
        "matrix_file": str(matrix.resolve()), "model_file": str(model_file.resolve()),
        "matrix_size_bytes": matrix.stat().st_size,
        "target_type_user_supplied": target, "chunksize": chunksize,
        "n_models": n, "n_gene_columns": p,
        "n_lineages_in_matrix_known_labels": int(rows.OncotreeLineage.nunique()),
        "n_models_missing_metadata": int((~rows.metadata_found).sum()),
        "n_models_missing_lineage": int(rows.OncotreeLineage.isna().sum()),
        "n_lineages_in_entire_model_csv_known_labels": int(metadata.OncotreeLineage.nunique()),
        "n_models_in_entire_model_csv": len(metadata),
        "n_observed_values": int(count.sum()),
        "pct_missing_values": float(100 * (n * p - count.sum()) / (n * p)),
        "n_genes_all_missing": int((count == 0).sum()),
        "n_genes_complete": int((count == n).sum()),
        "n_genes_at_least_95pct_observed": int((count / n >= .95).sum()),
        "n_models_all_targets_missing": int((rows.n_observed_genes == 0).sum()),
        "minimum_observed_score": float(minimum.min()) if count.sum() else None,
        "maximum_observed_score": float(maximum.max()) if count.sum() else None,
        "n_values_outside_0_1": outside_range,
        "warning": ("Values outside [0,1]: check target file/type before modelling"
                    if target == "dependency" and outside_range else None),
        "usage": "Descriptive full-cohort statistics only. Recompute baselines on training folds.",
    }
    out.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out / "gene_summary.csv", index=False)
    rows.to_csv(out / "model_summary.csv", index=False)
    lineage_counts.to_csv(out / "lineage_counts.csv", index=False)
    pd.concat(previews, ignore_index=True).to_csv(out / "small_preview.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Reports saved to {out.resolve()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, help="Original CRISPR CSV, not Excel re-export")
    parser.add_argument("--models", required=True, help="Model.csv from the SAME release")
    parser.add_argument("--out", default="depmap_audit")
    parser.add_argument("--chunksize", type=int, default=100)
    parser.add_argument("--target", choices=["dependency", "effect"], default="dependency")
    parser.add_argument("--release", default="26Q3", help="Label only; not independently verified")
    args = parser.parse_args()
    audit(args.matrix, args.models, args.out, args.chunksize, args.target, args.release)
