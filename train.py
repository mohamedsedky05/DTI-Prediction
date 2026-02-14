from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from dti.eval import (
    cross_validate_binary_classifier,
    extract_feature_importance,
    out_of_fold_pred_proba,
    save_json,
    save_roc_curve_plot,
)
from dti.features.dti import DTIFeaturizer, DTIFeaturizerConfig
from dti.features.drug import DrugFeaturizerConfig
from dti.features.protein import ProteinFeaturizerConfig
from dti.io import dataset_stats_to_dict, load_dti_csv
from dti.models import make_estimator, maybe_sample_weight_for_estimator
from dti.preprocess import SelectiveStandardScaler, continuous_mask_excluding_fingerprint
from dti.utils import set_global_seed


def build_pipeline(
    model_name: str,
    seed: int,
    kmer_k: int,
    kmer_dim: int,
    fp_bits: int,
) -> Pipeline:
    featurizer_cfg = DTIFeaturizerConfig(
        drug=DrugFeaturizerConfig(n_bits=fp_bits),
        protein=ProteinFeaturizerConfig(kmer_k=kmer_k, kmer_dim=kmer_dim),
    )
    featurize = DTIFeaturizer(cfg=featurizer_cfg)

    cont_mask = continuous_mask_excluding_fingerprint(
        n_features=featurize.n_features_,
        fingerprint_slice=featurize.fingerprint_slice_,
    )
    scale = SelectiveStandardScaler(continuous_mask=cont_mask)

    # estimator may depend on label imbalance (XGBoost scale_pos_weight)
    # so we construct it later if needed (after reading y).
    clf_placeholder = make_estimator("logreg", seed=seed, y_for_imbalance=None)

    return Pipeline(
        steps=[
            ("featurize", featurize),
            ("scale", scale),
            ("clf", clf_placeholder),
        ]
    )


def replace_classifier(pipeline: Pipeline, model_name: str, seed: int, y: np.ndarray) -> Pipeline:
    clf = make_estimator(model_name=model_name, seed=seed, y_for_imbalance=y)
    pipeline.set_params(clf=clf)
    return pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="DTI prediction from raw SMILES + protein sequence.")
    ap.add_argument("--data", required=True, help="Path to CSV with compound_iso_smiles,target_sequence,label.")
    ap.add_argument("--outdir", required=True, help="Output directory to save model/report.")
    ap.add_argument("--model", default="logreg", choices=["logreg", "rf", "xgb", "mlp"], help="Model type.")
    ap.add_argument("--cv", type=int, default=5, help="Stratified CV folds. Set 0 to skip CV.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--drop_invalid", action="store_true", help="Drop invalid SMILES/sequences (recommended).")
    ap.add_argument("--kmer_k", type=int, default=3, help="Protein k-mer length (for hashing).")
    ap.add_argument("--kmer_dim", type=int, default=1024, help="Protein hashed k-mer vector dimension.")
    ap.add_argument("--fp_bits", type=int, default=2048, help="Morgan fingerprint bit length.")
    ap.add_argument("--topk_importance", type=int, default=200, help="Top-K features to save (if available).")
    ap.add_argument(
        "--save_roc",
        action="store_true",
        help="Save out-of-fold ROC curve plot (only when --cv >= 2).",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, stats = load_dti_csv(args.data, drop_invalid=args.drop_invalid)
    y = df["label"].values.astype(int)

    # Build pipeline and plug in the chosen classifier (some need y for imbalance params).
    pipeline = build_pipeline(
        model_name=args.model,
        seed=args.seed,
        kmer_k=args.kmer_k,
        kmer_dim=args.kmer_dim,
        fp_bits=args.fp_bits,
    )
    pipeline = replace_classifier(pipeline, model_name=args.model, seed=args.seed, y=y)

    # Some estimators (MLP) benefit from sample weights for imbalance.
    sample_weight = maybe_sample_weight_for_estimator(pipeline.named_steps["clf"], y=y)

    report: Dict[str, Any] = {
        "data_path": str(args.data),
        "outdir": str(outdir),
        "model": args.model,
        "seed": args.seed,
        "drop_invalid": bool(args.drop_invalid),
        "kmer_k": int(args.kmer_k),
        "kmer_dim": int(args.kmer_dim),
        "fp_bits": int(args.fp_bits),
        "dataset_stats": dataset_stats_to_dict(stats),
        "n_rows_used": int(len(df)),
        "class_balance": {
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "pos_rate": float((y == 1).mean()) if len(y) else 0.0,
        },
        "cv": {},
    }

    # CV evaluation (leakage-safe via pipeline cloning).
    if args.cv and args.cv >= 2:
        folds, summary = cross_validate_binary_classifier(
            pipeline=pipeline,
            X=df[["compound_iso_smiles", "target_sequence"]],
            y=y,
            cv_folds=int(args.cv),
            seed=args.seed,
            sample_weight=sample_weight,
        )
        report["cv"] = summary

        if args.save_roc:
            y_proba_oof = out_of_fold_pred_proba(
                pipeline=pipeline,
                X=df[["compound_iso_smiles", "target_sequence"]],
                y=y,
                cv_folds=int(args.cv),
                seed=args.seed,
                sample_weight=sample_weight,
            )
            roc_path = outdir / "roc_curve_oof.png"
            roc_stats = save_roc_curve_plot(y_true=y, y_proba=y_proba_oof, path=str(roc_path))
            report["roc_curve_oof"] = {"path": str(roc_path), **roc_stats}

    # Fit final model on all data and save it.
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    pipeline.fit(df[["compound_iso_smiles", "target_sequence"]], y, **fit_kwargs)

    model_path = outdir / "model.joblib"
    joblib.dump(pipeline, model_path)

    # Feature importance (if available).
    imp_df = extract_feature_importance(pipeline)
    if imp_df is not None:
        imp_path = outdir / "feature_importance.csv"
        imp_df.head(int(args.topk_importance)).to_csv(imp_path, index=False)
        report["feature_importance"] = {
            "available": True,
            "path": str(imp_path),
            "method": str(imp_df["method"].iloc[0]) if len(imp_df) else None,
        }
    else:
        report["feature_importance"] = {"available": False}

    save_json(report, str(outdir / "report.json"))


if __name__ == "__main__":
    main()

