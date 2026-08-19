from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_project_directories
from .utils import environment_summary, get_logger


DOMAIN_COLORS = {
    "general": "#2E6FBB",
    "biomedical": "#B34747",
    "wikipedia_adapted": "#6A4C93",
    "control": "#777777",
    "baseline": "#222222",
}


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    selected = frame.reindex(columns=columns).copy()
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(
                lambda value: "" if not np.isfinite(value) else f"{value:.{digits}f}"
            )
        elif pd.api.types.is_object_dtype(selected[column]):
            nonempty = selected[column].astype(str).str.strip().ne("")
            numeric = pd.to_numeric(selected[column], errors="coerce")
            if nonempty.any() and numeric[nonempty].notna().all():
                selected[column] = numeric.map(
                    lambda value: "" if not np.isfinite(value) else f"{value:.{digits}f}"
                )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _decorate_summary(summary: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    decorated = summary.copy()
    if not sources.empty:
        decorated = decorated.merge(sources, on="embedding", how="left")
    for column, default in [
        ("display_name", ""),
        ("domain", "baseline"),
        ("text_variant", "baseline"),
        ("model_family", "zero"),
        ("adaptation", "none"),
    ]:
        if column not in decorated:
            decorated[column] = default
        decorated[column] = decorated[column].fillna(default).replace("", default)
    decorated.loc[decorated["method"] == "zero", "display_name"] = "Zero baseline"
    decorated.loc[decorated["method"] == "zero", "domain"] = "baseline"
    return decorated


def _save_figure(figure, project_paths: dict[str, Path], stem: str) -> None:
    figure.savefig(
        project_paths["figures"] / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        project_paths["figures"] / f"{stem}.pdf",
        bbox_inches="tight",
    )


def _performance_forest(project_paths: dict[str, Path], decorated: pd.DataFrame, plt) -> None:
    plot = decorated.sort_values("rmse_mean", ascending=True).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(12.5, max(6.0, 0.5 * len(plot) + 1.7)))
    positions = np.arange(len(plot))
    for index, row in plot.iterrows():
        low = max(float(row["rmse_mean"] - row["rmse_ci_low"]), 0.0)
        high = max(float(row["rmse_ci_high"] - row["rmse_mean"]), 0.0)
        axis.errorbar(
            float(row["rmse_mean"]),
            positions[index],
            xerr=np.asarray([[low], [high]]),
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=2,
            color=DOMAIN_COLORS.get(str(row["domain"]), "#555555"),
        )
    axis.set_yticks(positions, plot["display_name"])
    axis.invert_yaxis()
    axis.set_xlabel("Target-macro RMSE (95% target-bootstrap CI; lower is better)")
    axis.set_ylabel("")
    axis.set_title("Zero-shot performance using identical official NHANES descriptions")
    zero = plot.loc[plot["method"] == "zero", "rmse_mean"]
    if not zero.empty:
        axis.axvline(float(zero.iloc[0]), color="#222222", linestyle="--", linewidth=1.2)
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    _save_figure(figure, project_paths, "zero_shot_rmse_comparison")
    plt.close(figure)


def _adaptation_delta_figure(
    project_paths: dict[str, Path], paired_tasks: pd.DataFrame, paired_summary: pd.DataFrame, plt
) -> None:
    if paired_tasks.empty or "pair_type" not in paired_tasks:
        return
    selected = paired_tasks[paired_tasks["pair_type"] == "encoder_adaptation"].copy()
    if selected.empty:
        return
    target = (
        selected.groupby("target")["rmse_reference_minus_candidate"]
        .mean()
        .sort_values()
    )
    figure, axis = plt.subplots(figsize=(11.5, max(5.5, 0.35 * len(target) + 2.0)))
    colors = np.where(target.to_numpy() > 0, "#2E7D32", "#B34747")
    axis.scatter(target.to_numpy(), np.arange(len(target)), c=colors, s=55, zorder=3)
    axis.axvline(0.0, color="#222222", linewidth=1.2)
    axis.set_yticks(np.arange(len(target)), target.index)
    axis.set_xlabel("Base BGE RMSE − WikiMed-adapted BGE RMSE (positive favors adaptation)")
    axis.set_ylabel("Held-out target")
    axis.set_title("Paired target-level effect of external WikiMed encoder adaptation")
    summary = (
        paired_summary[paired_summary["pair_type"] == "encoder_adaptation"]
        if not paired_summary.empty and "pair_type" in paired_summary
        else pd.DataFrame()
    )
    if not summary.empty:
        row = summary.iloc[0]
        mean = float(row["rmse_improvement_mean"])
        low = float(row["rmse_improvement_ci_low"])
        high = float(row["rmse_improvement_ci_high"])
        axis.axvspan(low, high, color="#6A4C93", alpha=0.13)
        axis.axvline(mean, color="#6A4C93", linestyle="--", linewidth=2)
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    _save_figure(figure, project_paths, "wikimed_adaptation_paired_targets")
    plt.close(figure)


def _domain_performance_heatmap(
    project_paths: dict[str, Path],
    metrics: pd.DataFrame,
    tasks: pd.DataFrame,
    sources: pd.DataFrame,
    plt,
    sns,
) -> None:
    if "target_domain" not in tasks:
        return
    target_domains = tasks[["target", "target_domain"]].drop_duplicates("target")
    real_names = set(
        sources.loc[
            sources["domain"].isin(["general", "biomedical", "wikipedia_adapted"]),
            "embedding",
        ].astype(str)
    )
    per_target = metrics.groupby(["target", "method", "embedding"], as_index=False)["rmse"].mean()
    zero = per_target[per_target["method"] == "zero"].set_index("target")["rmse"]
    semantic = per_target[
        (per_target["method"].str.startswith("semantic_zero_shot__"))
        & per_target["embedding"].isin(real_names)
    ].copy()
    if semantic.empty:
        return
    semantic["improvement"] = semantic.apply(
        lambda row: float(zero.get(str(row["target"]), np.nan)) - float(row["rmse"]), axis=1
    )
    semantic = semantic.merge(target_domains, on="target", how="left")
    values = semantic.groupby(["target_domain", "embedding"])["improvement"].mean().unstack()
    values = values.rename(columns=sources.set_index("embedding")["display_name"].to_dict())
    magnitude = max(float(np.nanmax(np.abs(values.to_numpy()))), 0.05)
    figure, axis = plt.subplots(
        figsize=(max(11.5, 1.15 * len(values.columns) + 4), max(6.0, 0.55 * len(values) + 2))
    )
    sns.heatmap(
        values,
        cmap="vlag",
        center=0,
        vmin=-magnitude,
        vmax=magnitude,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "RMSE improvement over zero (positive is better)"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Held-out target domain")
    axis.set_title("Zero-shot improvement by target domain")
    axis.tick_params(axis="x", rotation=45, labelsize=9)
    axis.tick_params(axis="y", rotation=0, labelsize=9)
    figure.tight_layout()
    _save_figure(figure, project_paths, "target_domain_improvement_heatmap")
    plt.close(figure)


def _rank_selection_figure(project_paths: dict[str, Path], ranks: pd.DataFrame, plt) -> None:
    if ranks.empty:
        return
    ranks = ranks.sort_values("rank")
    error = pd.to_numeric(ranks.get("validation_rmse_sd", 0), errors="coerce").fillna(0)
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.errorbar(
        ranks["rank"],
        ranks["validation_rmse_mean"],
        yerr=error,
        marker="o",
        capsize=5,
        linewidth=2,
        color="#2E6FBB",
    )
    best = ranks.sort_values(["validation_rmse_mean", "rank"]).iloc[0]
    axis.scatter([best["rank"]], [best["validation_rmse_mean"]], s=125, color="#6A4C93", zorder=4)
    axis.set_xticks(ranks["rank"].astype(int))
    axis.set_xlabel("Bilinear operator rank")
    axis.set_ylabel("Held-target validation macro-RMSE")
    axis.set_title("Validation-only global rank selection")
    figure.tight_layout()
    _save_figure(figure, project_paths, "rank_selection_validation")
    plt.close(figure)


def _cross_domain_figure(project_paths: dict[str, Path], matrix: pd.DataFrame, plt, sns) -> None:
    if matrix.empty:
        return
    values = matrix.groupby(["target_domain", "feature_domain"])["n_tasks"].sum().unstack(fill_value=0)
    figure, axis = plt.subplots(
        figsize=(max(10.0, 0.55 * len(values.columns) + 4), max(7.0, 0.5 * len(values) + 2))
    )
    sns.heatmap(
        values,
        cmap="Blues",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Qualified tasks"},
        ax=axis,
    )
    axis.set_xlabel("Predictor domain")
    axis.set_ylabel("Target domain")
    axis.set_title("Audited cross-domain benchmark coverage")
    axis.tick_params(axis="x", rotation=50, labelsize=8)
    axis.tick_params(axis="y", rotation=0, labelsize=8)
    figure.tight_layout()
    _save_figure(figure, project_paths, "cross_domain_task_coverage")
    plt.close(figure)


def _make_figures(
    config: dict[str, Any],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    tasks: pd.DataFrame,
    sources: pd.DataFrame,
    paired_tasks: pd.DataFrame,
    paired_summary: pd.DataFrame,
    ranks: pd.DataFrame,
    domain_matrix: pd.DataFrame,
) -> None:
    project_paths = ensure_project_directories(config)
    mpl_dir = project_paths["outputs"] / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    _performance_forest(project_paths, _decorate_summary(summary, sources), plt)
    _adaptation_delta_figure(project_paths, paired_tasks, paired_summary, plt)
    _domain_performance_heatmap(project_paths, metrics, tasks, sources, plt, sns)
    _rank_selection_figure(project_paths, ranks, plt)
    _cross_domain_figure(project_paths, domain_matrix, plt, sns)


def _embedding_source_table(sources: pd.DataFrame) -> pd.DataFrame:
    table = sources.copy()
    source_values = table["source_url"] if "source_url" in table else pd.Series("", index=table.index)
    corpus_values = (
        table["training_corpus_url"]
        if "training_corpus_url" in table
        else pd.Series("", index=table.index)
    )
    table["source"] = source_values.map(
        lambda url: "generated locally" if url == "generated-locally" else f"[model card]({url})"
    )
    table["training corpus"] = corpus_values.map(
        lambda url: "" if not str(url).strip() else f"[fixed WikiMed ZIM]({url})"
    )
    return table.rename(
        columns={
            "display_name": "embedding variant",
            "text_variant": "downstream text",
            "model_id": "model ID",
            "base_model_id": "base model",
        }
    )


def _figure_line(project_paths: dict[str, Path], filename: str, alt: str) -> list[str]:
    return [f"![{alt}](figures/{filename})", ""] if (project_paths["figures"] / filename).exists() else []


def generate_report(config: dict[str, Any], force: bool = False) -> Path:
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    report_path = project_paths["outputs"] / "report.md"
    if report_path.exists() and not force:
        logger.info("Reusing %s", report_path)
        return report_path
    required = {
        "summary": project_paths["metrics"] / "summary_metrics.csv",
        "metrics": project_paths["metrics"] / "per_task_metrics.csv",
        "tasks": project_paths["tasks"] / "tasks.csv",
        "sources": project_paths["embeddings"] / "embedding_sources.csv",
    }
    if not all(path.exists() for path in required.values()):
        raise FileNotFoundError("Run tasks, adapt, embed, rank, train, and evaluate before report")

    summary = pd.read_csv(required["summary"]).fillna("")
    metrics = pd.read_csv(required["metrics"]).fillna("")
    tasks = pd.read_csv(required["tasks"]).fillna("")
    sources = pd.read_csv(required["sources"]).fillna("")
    paired_tasks = _read_optional_csv(project_paths["metrics"] / "paired_zero_shot_task_deltas.csv")
    paired_summary = _read_optional_csv(project_paths["metrics"] / "paired_zero_shot_summary.csv")
    ranks = _read_optional_csv(
        project_paths["checkpoints"] / "rank_selection" / "rank_selection_summary.csv"
    )
    domain_matrix = _read_optional_csv(project_paths["tasks"] / "cross_domain_task_matrix.csv")
    file_manifest = _read_optional_csv(project_paths["audit"] / "nhanes_file_manifest.csv")
    table_granularity = _read_optional_csv(
        project_paths["audit"] / "nhanes_table_granularity.csv"
    )
    selected_rank = _read_optional_json(project_paths["audit"] / "selected_operator_rank.json")
    adaptation = _read_optional_json(project_paths["audit"] / "wikimed_adaptation_manifest.json")

    _make_figures(
        config,
        summary,
        metrics,
        tasks,
        sources,
        paired_tasks,
        paired_summary,
        ranks,
        domain_matrix,
    )
    decorated = _decorate_summary(summary, sources)
    learned = decorated[
        (decorated["method_family"] == "semantic_zero_shot")
        & ~decorated["domain"].isin(["control", "baseline"])
    ]
    best = learned.sort_values("rmse_mean").iloc[0] if not learned.empty else None
    task_split = (
        tasks.groupby("task_split")
        .agg(tasks=("task_id", "count"), targets=("target", "nunique"))
        .reset_index()
    )
    target_domain = (
        tasks.groupby(["task_split", "target_domain"])
        .agg(tasks=("task_id", "count"), targets=("target", "nunique"))
        .reset_index()
        if "target_domain" in tasks
        else pd.DataFrame()
    )
    env = environment_summary()
    rank = int(selected_rank.get("selected_rank", config["training"]["rank"]))

    lines = ["# NHANES cross-domain zero-shot semantic regression report", "", "## Executive result", ""]
    if best is not None:
        lines.append(
            f"The best learned encoder was **{best['display_name']}**, with target-macro RMSE "
            f"{float(best['rmse_mean']):.4f} and MAE {float(best['mae_mean']):.4f} on "
            "target-held-out tasks and participant-held-out test rows."
        )
    else:
        lines.append("No learned encoder result was available.")
    lines.extend(
        [
            "",
            "Every learned encoder receives the identical official NHANES variable description. Wikipedia text is used only to adapt one BGE encoder before the NHANES experiment; it is never appended to a variable description. The held-out-task comparison is purely zero-shot, with no calibration labels or task-specific regression fit.",
            "",
        ]
    )
    lines += _figure_line(project_paths, "zero_shot_rmse_comparison.png", "Zero-shot RMSE comparison")
    lines.extend(
        [
            "## Zero-shot performance",
            "",
            _markdown_table(
                decorated.rename(columns={"display_name": "variant"}),
                [
                    "variant",
                    "domain",
                    "adaptation",
                    "n_tasks",
                    "n_targets",
                    "rmse_mean",
                    "rmse_ci_low",
                    "rmse_ci_high",
                    "mae_mean",
                    "r2_mean",
                    "pearson_mean",
                ],
            ),
            "",
            "Tasks are averaged within held-out target before targets are averaged equally. Confidence intervals bootstrap targets, not participant rows.",
            "",
            "## External WikiMed encoder adaptation",
            "",
            "The WikiMed comparator starts from BGE base and is contrastively adapted on deterministic title-to-lead and adjacent-paragraph pairs extracted from the fixed local medical Wikipedia snapshot. Adaptation never reads NHANES text, participant values, task qualification scores, or benchmark outcomes. After adaptation, the encoder is frozen.",
            "",
        ]
    )
    lines += _figure_line(
        project_paths,
        "wikimed_adaptation_paired_targets.png",
        "Paired target effects of WikiMed adaptation",
    )
    adaptation_pairs = (
        paired_summary[paired_summary["pair_type"] == "encoder_adaptation"]
        if not paired_summary.empty and "pair_type" in paired_summary
        else pd.DataFrame()
    )
    if not adaptation_pairs.empty:
        lines.extend(
            [
                _markdown_table(
                    adaptation_pairs,
                    [
                        "n_tasks",
                        "n_targets",
                        "rmse_improvement_mean",
                        "rmse_improvement_ci_low",
                        "rmse_improvement_ci_high",
                        "candidate_target_win_rate",
                    ],
                ),
                "",
            ]
        )
    if adaptation:
        retrieval = adaptation.get("validation_retrieval", {})
        base = retrieval.get("base", {})
        adapted = retrieval.get("adapted", {})
        if base and adapted:
            lines.extend(
                [
                    f"On the held-out external-corpus retrieval diagnostic, Recall@1 changed from {float(base.get('recall_at_1', float('nan'))):.4f} to {float(adapted.get('recall_at_1', float('nan'))):.4f}. This diagnostic validates the adaptation objective; NHANES test performance remains the substantive endpoint.",
                    "",
                ]
            )

    lines.extend(["## Validation-only rank selection", ""])
    lines += _figure_line(project_paths, "rank_selection_validation.png", "Operator rank selection")
    if not ranks.empty:
        lines.extend(
            [
                _markdown_table(ranks.sort_values("rank"), ["rank", "validation_rmse_mean", "validation_rmse_sd"]),
                "",
            ]
        )
    lines.extend(
        [
            f"The globally applied bilinear rank is **{rank}**. It was selected using target-macro RMSE on held-out validation targets across the prespecified candidates; test targets were not consulted.",
            "",
            "## Benchmark construction and domain coverage",
            "",
            _markdown_table(task_split, ["task_split", "tasks", "targets"], digits=0),
            "",
        ]
    )
    if not target_domain.empty:
        lines.extend(
            [
                _markdown_table(target_domain, ["task_split", "target_domain", "tasks", "targets"], digits=0),
                "",
            ]
        )
    lines += _figure_line(project_paths, "cross_domain_task_coverage.png", "Cross-domain task coverage")
    lines.extend(
        [
            f"A task is retained only when a factory-participant Ridge model reaches qualification R2 >= {config['task_factory']['min_qualification_r2']} and improves RMSE over the discovery mean by at least {100 * float(config['task_factory']['min_rmse_improvement_fraction']):.1f}% on separate qualification participants. Predictors must span at least {config['task_factory']['min_feature_domains']} semantic domains and {config['task_factory']['min_feature_tables']} source tables, and no predictor may share the target domain. Targets are disjoint across task splits.",
            "",
        ]
    )
    if not file_manifest.empty:
        coverage = (
            file_manifest.groupby("collection_component")["file_id"]
            .nunique()
            .rename("public tables")
            .reset_index()
        )
        lines.extend(
            [
                "The input manifest is locked on first resolution from all public 2017–2018 NHANES Demographics, Dietary, Examination, Laboratory, and Questionnaire pages:",
                "",
                _markdown_table(coverage, ["collection_component", "public tables"], digits=0),
                "",
            ]
        )
    if not table_granularity.empty and "merge_action" in table_granularity:
        granularity_summary = (
            table_granularity.groupby(["row_granularity", "merge_action"])["file_id"]
            .nunique()
            .rename("tables")
            .reset_index()
        )
        lines.extend(
            [
                "Every public table is profiled before merging. Repeated-record and non-participant reference tables are audited but excluded from the one-row-per-participant matrix; no arbitrary first record, undocumented mean, or lookup-code participant is used:",
                "",
                _markdown_table(
                    granularity_summary,
                    ["row_granularity", "merge_action", "tables"],
                    digits=0,
                ),
                "",
            ]
        )
    lines += _figure_line(
        project_paths,
        "target_domain_improvement_heatmap.png",
        "Embedding improvement by target domain",
    )

    lines.extend(
        [
            "## Model",
            "",
            "For each task, factory-standardized predictor values multiply frozen official-description embeddings. Their square-root-normalized sum passes through one shared low-rank bilinear operator and is paired with the frozen target embedding. Training uses task-balanced row weights; early stopping and rank selection use target-macro validation RMSE.",
            "",
            "## Embedding sources",
            "",
            _markdown_table(
                _embedding_source_table(sources),
                [
                    "embedding variant",
                    "domain",
                    "adaptation",
                    "dimension",
                    "model ID",
                    "base model",
                    "source",
                    "training corpus",
                    "license",
                ],
                digits=0,
            ),
            "",
            f"NHANES variable metadata comes from the [NCHS documentation](https://wwwn.cdc.gov/nchs/nhanes/). The external adaptation corpus is the fixed [Kiwix English medicine mini snapshot]({config['wikipedia']['zim_url']}) read locally with [OpenZIM's Python reader](https://github.com/openzim/python-libzim). The adaptation objective is Sentence Transformers' cached multiple-negatives ranking loss; no page-level API requests are made.",
            "",
            "## Leakage and interpretation boundaries",
            "",
            "- `SEQN`, weights, survey-design fields, statuses, comments, documented missing codes, same-concept repeats, derived groups, and near-affine duplicates are unavailable as predictors.",
            "- Missing-value sentinels documented in each NHANES codebook are converted to missing before eligibility and normalization.",
            "- XPORT character fields are decoded with a prespecified table-wide fallback order, and the selected encoding is recorded in the table audit.",
            "- Public repeated-record and non-participant reference tables are retained in the audits but excluded from the participant-level matrix; their variables cannot become task predictors or targets.",
            "- Eligibility, normalization, task discovery, and Ridge qualification use factory participants only. Operator fitting uses training participants; early stopping uses validation targets and participants; final scoring uses test targets and participants.",
            "- The external WikiMed corpus split is article-disjoint and contains no NHANES metadata or benchmark labels. The adapted encoder is frozen before NHANES embeddings are generated.",
            "- Held-out targets may have appeared as predictors in meta-training tasks. This is target-role transfer, not strict vocabulary exclusion.",
            "- The task factory reads factory outcomes to retain linearly feasible tasks; it does not claim outcome-label-free task discovery.",
            "- NHANES survey weights are intentionally unused because this is predictive machine learning, not population or causal inference.",
            "",
            "## Reproducibility",
            "",
            f"- Seed: `{config['project']['seed']}`",
            f"- Device: `{env.get('cuda_devices') or 'CPU / unavailable'}`",
            f"- PyTorch: `{env.get('torch')}`; CUDA runtime: `{env.get('cuda_runtime')}`",
            f"- Aggregation: `{config['training']['aggregation']}`; validation-selected rank: `{rank}`",
            f"- Target-bootstrap repetitions: `{config['evaluation']['bootstrap_repetitions']}`",
            f"- WikiMed snapshot: `{config['wikipedia']['zim_path']}`",
            "- Resolved configuration, locked table manifest, row-granularity and missing-code audits, domain assignments, task/leakage audits, adaptation corpus manifest, embeddings, checkpoints, predictions, metrics, and figures are stored beside this report.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return report_path
