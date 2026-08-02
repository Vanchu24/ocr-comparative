from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ENGINE_ORDER = ["PaddleOCR", "EasyOCR", "Doctr"]
DISPLAY_NAME = {
    "Doctr": "DocTR",
    "PaddleOCR": "PaddleOCR",
    "EasyOCR": "EasyOCR",
}


def read_csv_checked(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Read a CSV and verify that all required columns are present."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found:\n{path}\n\n"
            "Check the input_data folder described in the README."
        )

    df = pd.read_csv(path)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return df


def sort_engines(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same engine order to every table and figure."""
    order = {engine: index for index, engine in enumerate(ENGINE_ORDER)}
    result = df.copy()
    result["_engine_order"] = result["ocr_engine"].map(order).fillna(999)
    return (
        result.sort_values(["_engine_order", "ocr_engine"])
        .drop(columns="_engine_order")
        .reset_index(drop=True)
    )


def add_display_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ocr_engine_display"] = (
        result["ocr_engine"].map(DISPLAY_NAME).fillna(result["ocr_engine"])
    )
    return result


def select_best_variant(by_variant: pd.DataFrame) -> pd.DataFrame:
    """
    Select one configuration per OCR engine.

    Primary criterion: lowest average CER
    Tie-breakers: lowest average WER, then shortest processing time
    """
    ranked = by_variant.sort_values(
        ["ocr_engine", "average_cer", "average_wer", "average_time_seconds"]
    )
    best = ranked.groupby("ocr_engine", as_index=False).first()
    return add_display_names(sort_engines(best))


def export_tables(
    overall: pd.DataFrame,
    by_variant: pd.DataFrame,
    best: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export poster tables as CSV and Excel files."""
    overall_export = overall[
        [
            "ocr_engine_display",
            "documents_evaluated",
            "documents_successful",
            "average_cer",
            "average_wer",
            "average_time_seconds",
            "average_reference_token_recall",
            "success_rate_percent",
        ]
    ].rename(
        columns={
            "ocr_engine_display": "OCR Engine",
            "documents_evaluated": "Runs Evaluated",
            "documents_successful": "Successful Runs",
            "average_cer": "Average CER",
            "average_wer": "Average WER",
            "average_time_seconds": "Average Processing Time (s)",
            "average_reference_token_recall": "Reference Token Recall",
            "success_rate_percent": "Success Rate (%)",
        }
    )

    by_variant_export = by_variant[
        [
            "ocr_engine_display",
            "preprocessing_variant",
            "documents_evaluated",
            "documents_successful",
            "average_cer",
            "average_wer",
            "average_time_seconds",
            "average_reference_token_recall",
            "success_rate_percent",
        ]
    ].rename(
        columns={
            "ocr_engine_display": "OCR Engine",
            "preprocessing_variant": "Preprocessing Variant",
            "documents_evaluated": "Documents Evaluated",
            "documents_successful": "Successful Documents",
            "average_cer": "Average CER",
            "average_wer": "Average WER",
            "average_time_seconds": "Average Processing Time (s)",
            "average_reference_token_recall": "Reference Token Recall",
            "success_rate_percent": "Success Rate (%)",
        }
    )

    best_export = best[
        [
            "ocr_engine_display",
            "preprocessing_variant",
            "documents_evaluated",
            "documents_successful",
            "average_cer",
            "average_wer",
            "average_time_seconds",
            "average_reference_token_recall",
            "success_rate_percent",
        ]
    ].rename(
        columns={
            "ocr_engine_display": "OCR Engine",
            "preprocessing_variant": "Selected Variant",
            "documents_evaluated": "Documents Evaluated",
            "documents_successful": "Successful Documents",
            "average_cer": "Average CER",
            "average_wer": "Average WER",
            "average_time_seconds": "Average Processing Time (s)",
            "average_reference_token_recall": "Reference Token Recall",
            "success_rate_percent": "Success Rate (%)",
        }
    )

    overall_export.to_csv(
        output_dir / "engine_aggregate_all_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    by_variant_export.to_csv(
        output_dir / "engine_by_preprocessing.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_export.to_csv(
        output_dir / "comparative_performance_best_variant.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(output_dir / "poster_ocr_tables.xlsx") as writer:
        best_export.to_excel(writer, sheet_name="Main Comparison", index=False)
        by_variant_export.to_excel(writer, sheet_name="By Preprocessing", index=False)
        overall_export.to_excel(writer, sheet_name="All Runs", index=False)


def label_bars(ax, bars, decimals: int = 3, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        if np.isfinite(value):
            ax.annotate(
                f"{value:.{decimals}f}{suffix}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def save_accuracy_chart(best: pd.DataFrame, output_dir: Path) -> None:
    labels = best["ocr_engine_display"].tolist()
    cer = best["average_cer"].to_numpy(float)
    wer = best["average_wer"].to_numpy(float)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    cer_bars = ax.bar(x - width / 2, cer, width, label="CER")
    wer_bars = ax.bar(x + width / 2, wer, width, label="WER")

    ax.set_title("Recognition Error by OCR Engine")
    ax.set_xlabel("OCR engine")
    ax.set_ylabel("Average error rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    label_bars(ax, cer_bars)
    label_bars(ax, wer_bars)

    fig.tight_layout()
    fig.savefig(output_dir / "cer_wer_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_processing_time_chart(best: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        best["ocr_engine_display"],
        best["average_time_seconds"].to_numpy(float),
    )

    ax.set_title("Average Document Processing Time")
    ax.set_xlabel("OCR engine")
    ax.set_ylabel("Average time (seconds)")
    ax.grid(axis="y", alpha=0.25)
    label_bars(ax, bars, decimals=2, suffix=" s")

    fig.tight_layout()
    fig.savefig(output_dir / "processing_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_preprocessing_chart(by_variant: pd.DataFrame, output_dir: Path) -> None:
    pivot = by_variant.pivot(
        index="ocr_engine_display",
        columns="preprocessing_variant",
        values="average_cer",
    )
    desired_order = ["PaddleOCR", "EasyOCR", "DocTR"]
    pivot = pivot.reindex([name for name in desired_order if name in pivot.index])

    variants = [name for name in ["none", "enhanced"] if name in pivot.columns]
    x = np.arange(len(pivot.index))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = [-width / 2, width / 2]

    for variant, offset in zip(variants, offsets):
        bars = ax.bar(
            x + offset,
            pivot[variant].to_numpy(float),
            width,
            label=variant.capitalize(),
        )
        label_bars(ax, bars)

    ax.set_title("Effect of Image Preprocessing on Character Error Rate")
    ax.set_xlabel("OCR engine")
    ax.set_ylabel("Average CER")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.legend(title="Input variant")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "preprocessing_effect_cer.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_distribution_chart(
    detail: pd.DataFrame,
    best: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected = detail.merge(
        best[["ocr_engine", "preprocessing_variant"]],
        on=["ocr_engine", "preprocessing_variant"],
        how="inner",
    )

    labels, values = [], []
    for engine in ENGINE_ORDER:
        rows = selected[selected["ocr_engine"] == engine]
        if rows.empty:
            continue
        labels.append(DISPLAY_NAME.get(engine, engine))
        values.append(rows["cer"].dropna().to_numpy(float))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_title("Document-Level CER Distribution")
    ax.set_xlabel("OCR engine")
    ax.set_ylabel("Character Error Rate")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_dir / "document_level_cer_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def wrap_text(value: object, width: int = 65, limit: int = 650) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return "\n".join(textwrap.wrap(text, width=width))


def choose_example(detail: pd.DataFrame, best: pd.DataFrame) -> str:
    selected = detail.merge(
        best[["ocr_engine", "preprocessing_variant"]],
        on=["ocr_engine", "preprocessing_variant"],
        how="inner",
    )
    pivot = selected.pivot_table(
        index="image_id",
        columns="ocr_engine",
        values="wer",
        aggfunc="mean",
    )
    pivot["spread"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot = pivot.sort_values("spread", ascending=False)
    return str(pivot.index[1] if len(pivot) > 1 else pivot.index[0])


def save_qualitative_example(
    detail: pd.DataFrame,
    best: pd.DataFrame,
    repository_root: Path,
    output_dir: Path,
    requested_image: str | None,
) -> None:
    image_id = requested_image or choose_example(detail, best)
    image_id = str(image_id).replace(".jpg", "")

    selected = detail.merge(
        best[["ocr_engine", "preprocessing_variant"]],
        on=["ocr_engine", "preprocessing_variant"],
        how="inner",
    )
    subset = selected[selected["image_id"].astype(str) == image_id].copy()

    if subset.empty:
        print(f"Qualitative figure skipped: image ID {image_id} was not found.")
        return

    image_path = repository_root / "sample_image" / f"{image_id}.jpg"
    if not image_path.exists():
        print(
            "Qualitative figure skipped because the optional sample image was not "
            f"found at {image_path}."
        )
        return

    subset = sort_engines(subset)
    image = Image.open(image_path).convert("RGB")

    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(5, 1, height_ratios=[2.4, 0.7, 1, 1, 1])

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    ax.set_title(f"Original clinical document: {image_id}.jpg")
    ax.axis("off")

    reference = subset["reference_text"].iloc[0]
    ax = fig.add_subplot(grid[1, 0])
    ax.axis("off")
    ax.text(
        0,
        1,
        "Ground truth\n" + wrap_text(reference),
        ha="left",
        va="top",
        family="monospace",
    )

    for index, (_, row) in enumerate(subset.iterrows(), start=2):
        ax = fig.add_subplot(grid[index, 0])
        ax.axis("off")
        engine = DISPLAY_NAME.get(row["ocr_engine"], row["ocr_engine"])
        title = (
            f"{engine} — {row['preprocessing_variant']} "
            f"(CER={row['cer']:.3f}, WER={row['wer']:.3f})"
        )
        ax.text(
            0,
            1,
            title + "\n" + wrap_text(row["ocr_text"]),
            ha="left",
            va="top",
            family="monospace",
        )

    fig.suptitle("Representative Comparison Between Ground Truth and OCR Outputs")
    fig.tight_layout()
    fig.savefig(
        output_dir / "qualitative_ocr_comparison.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def write_summary(best: pd.DataFrame, output_dir: Path) -> None:
    lowest_cer = best.loc[best["average_cer"].idxmin()]
    lowest_wer = best.loc[best["average_wer"].idxmin()]
    fastest = best.loc[best["average_time_seconds"].idxmin()]
    highest_recall = best.loc[best["average_reference_token_recall"].idxmax()]

    text = f"""RESULT SUMMARY
==============

Best tested configuration per engine:
{chr(10).join(
    f"- {row.ocr_engine_display}: {row.preprocessing_variant} "
    f"(CER {row.average_cer:.4f}, WER {row.average_wer:.4f}, "
    f"time {row.average_time_seconds:.4f} s)"
    for row in best.itertuples()
)}

Main findings:
- Lowest CER: {lowest_cer['ocr_engine_display']} ({lowest_cer['average_cer']:.4f})
- Lowest WER: {lowest_wer['ocr_engine_display']} ({lowest_wer['average_wer']:.4f})
- Fastest engine: {fastest['ocr_engine_display']} ({fastest['average_time_seconds']:.4f} s)
- Highest reference-token recall: {highest_recall['ocr_engine_display']} ({highest_recall['average_reference_token_recall']:.4f})

Interpretation:
CER and WER may exceed 1.0 when insertions, substitutions, and deletions together
exceed the number of reference characters or words.
"""
    (output_dir / "results_summary.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OCR poster results from local exported CSV files."
    )
    parser.add_argument(
        "--example-image",
        type=str,
        default=None,
        help='Optional image ID for the qualitative figure, for example "5".',
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parent
    input_dir = repository_root / "input_data"
    output_dir = repository_root / "generated_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = read_csv_checked(
        input_dir / "ocr_ground_truth_by_engine.csv",
        {
            "ocr_engine",
            "documents_evaluated",
            "documents_successful",
            "average_cer",
            "average_wer",
            "average_time_seconds",
            "average_reference_token_recall",
            "success_rate_percent",
        },
    )
    by_variant = read_csv_checked(
        input_dir / "ocr_ground_truth_by_engine_preprocessing.csv",
        {
            "ocr_engine",
            "preprocessing_variant",
            "documents_evaluated",
            "documents_successful",
            "average_cer",
            "average_wer",
            "average_time_seconds",
            "average_reference_token_recall",
            "success_rate_percent",
        },
    )
    detail = read_csv_checked(
        input_dir / "ocr_ground_truth_detail_public.csv",
        {
            "image_id",
            "ocr_engine",
            "preprocessing_variant",
            "cer",
            "wer",
            "reference_token_recall",
        },
    )

    # The public detail file may omit clinical text. These columns are needed only
    # for the optional qualitative figure.
    for optional_column in ["reference_text", "ocr_text"]:
        if optional_column not in detail.columns:
            detail[optional_column] = ""

    overall = add_display_names(sort_engines(overall))
    by_variant = add_display_names(sort_engines(by_variant))
    best = select_best_variant(by_variant)

    export_tables(overall, by_variant, best, output_dir)
    save_accuracy_chart(best, output_dir)
    save_processing_time_chart(best, output_dir)
    save_preprocessing_chart(by_variant, output_dir)
    save_distribution_chart(detail, best, output_dir)

    if "reference_text" in detail.columns and "ocr_text" in detail.columns:
        save_qualitative_example(
            detail,
            best,
            repository_root,
            output_dir,
            args.example_image,
        )

    write_summary(best, output_dir)

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        print(f" - {path.relative_to(repository_root)}")


if __name__ == "__main__":
    main()