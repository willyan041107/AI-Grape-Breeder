from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pdfplumber


TABLE_TITLE_RE = re.compile(
    r"Table\.?\s*S\.?\s*(\d+)\.?",
    re.IGNORECASE,
)

NUM_RE = re.compile(
    r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
)


def extract_page_texts(pdf_path: Path) -> list[str]:
    """
    Extract text from every PDF page.
    """
    with pdfplumber.open(pdf_path) as pdf:
        return [
            page.extract_text(
                x_tolerance=2,
                y_tolerance=3,
            ) or ""
            for page in pdf.pages
        ]


def find_table_starts(
    page_texts: list[str],
) -> dict[int, int]:
    """
    Map supplementary table number -> 0-based starting page.

    Page 1 is skipped because it contains only the
    supplementary table index.
    """
    starts: dict[int, int] = {}

    for page_idx, text in enumerate(page_texts):

        if page_idx == 0:
            continue

        for match in TABLE_TITLE_RE.finditer(text):
            table_num = int(match.group(1))

            if table_num not in starts:
                starts[table_num] = page_idx

    return starts


def get_table_block(
    page_texts: list[str],
    starts: dict[int, int],
    table_num: int,
) -> str:
    """
    Return text from Table S{table_num} until the next
    supplementary table begins.
    """
    if table_num not in starts:
        raise ValueError(
            f"Could not locate Table S{table_num}"
        )

    start = starts[table_num]

    later_pages = [
        page
        for _, page in starts.items()
        if page > start
    ]

    end = (
        min(later_pages)
        if later_pages
        else len(page_texts)
    )

    return "\n".join(
        page_texts[start:end]
    )


def extract_s1(
    page_texts: list[str],
    starts: dict[int, int],
    outdir: Path,
) -> pd.DataFrame:

    text = get_table_block(
        page_texts,
        starts,
        1,
    )

    models = [
        "AlexNet",
        "VGG16",
        "ResNet50",
        "ResNet101",
        "InceptionV3",
        "DenseNet121",
    ]

    cv_rows: dict[str, dict[str, object]] = {}
    test_rows: dict[str, tuple[float, float]] = {}

    in_test_section = False

    for raw_line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            raw_line.strip(),
        )

        if not line:
            continue

        if line.lower().startswith("test sets"):
            in_test_section = True
            continue

        for model in models:

            if not line.startswith(model + " "):
                continue

            parts = line.split()

            if in_test_section:

                if len(parts) >= 3:
                    test_rows[model] = (
                        float(parts[1]),
                        float(parts[2]),
                    )

            else:

                if len(parts) < 18:
                    raise ValueError(
                        f"Unexpected S1 row:\n{line}"
                    )

                accuracy_values = list(
                    map(float, parts[1:8])
                )

                accuracy_group = parts[8]

                f1_values = list(
                    map(float, parts[9:16])
                )

                f1_group = parts[16]

                model_complexity = int(
                    float(parts[17])
                )

                row: dict[str, object] = {
                    "model": model,
                }

                for i in range(5):
                    row[f"accuracy_cv{i + 1}"] = (
                        accuracy_values[i]
                    )

                row["accuracy_mean"] = (
                    accuracy_values[5]
                )

                row["accuracy_sd"] = (
                    accuracy_values[6]
                )

                row["accuracy_group"] = (
                    accuracy_group
                )

                for i in range(5):
                    row[f"f1_cv{i + 1}"] = (
                        f1_values[i]
                    )

                row["f1_mean"] = f1_values[5]
                row["f1_sd"] = f1_values[6]
                row["f1_group"] = f1_group

                row["model_complexity"] = (
                    model_complexity
                )

                cv_rows[model] = row

            break

    output_rows = []

    for model in models:

        if model not in cv_rows:
            raise ValueError(
                f"Missing CV row for {model}"
            )

        if model not in test_rows:
            raise ValueError(
                f"Missing test row for {model}"
            )

        row = cv_rows[model].copy()

        test_accuracy, test_f1 = (
            test_rows[model]
        )

        row["test_accuracy"] = test_accuracy
        row["test_f1"] = test_f1

        output_rows.append(row)

    df = pd.DataFrame(output_rows)

    output_path = (
        outdir / "cnn_model_performance.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return df


def extract_s3(
    page_texts: list[str],
    starts: dict[int, int],
    outdir: Path,
) -> pd.DataFrame:

    text = get_table_block(
        page_texts,
        starts,
        3,
    )

    row_pattern = re.compile(
        r"^(VV_\d+)\s+"
        r"(table|wine)\s+"
        r"(Europe and America|Europe and Asia)\s+"
        r"(mild|severe)\((0|1)\)\s+"
        r"([0-9]+(?:\.[0-9]+)?)$",
        re.IGNORECASE,
    )

    rows = []

    for raw_line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            raw_line.strip(),
        )

        match = row_pattern.match(line)

        if not match:
            continue

        (
            accession_id,
            usage,
            population_group,
            binary_label,
            binary_code,
            continuous_phenotype,
        ) = match.groups()

        usage = usage.lower()

        rows.append(
            {
                "accession_id": accession_id,
                "usage": usage,
                "population_group": population_group,
                "binary_label": binary_label.lower(),
                "binary_code": int(binary_code),
                "continuous_phenotype": float(
                    continuous_phenotype
                ),
            }
        )

    df = pd.DataFrame(rows)

    df = (
        df.drop_duplicates(
            subset=["accession_id"]
        )
        .sort_values("accession_id")
        .reset_index(drop=True)
    )

    if len(df) != 231:
        raise ValueError(
            "Table S3 validation failed: "
            f"expected 231 accessions, got {len(df)}"
        )

    output_path = (
        outdir / "grapevine_phenotypes.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return df

def extract_cv_grid(
    page_texts: list[str],
    starts: dict[int, int],
    table_num: int,
    snp_counts: list[str],
    models: list[str],
    output_name: str,
    outdir: Path,
) -> pd.DataFrame:

    text = get_table_block(
        page_texts,
        starts,
        table_num,
    )

    expected_scores = (
        len(snp_counts) * len(models)
    )

    rows: list[dict[str, object]] = []

    for raw_line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            raw_line.strip(),
        )

        # CV rows begin with fold number
        if not re.match(r"^\d+\s", line):
            continue

        numeric_tokens = NUM_RE.findall(line)

        # First number = CV fold
        # Remaining numbers = scores
        if len(numeric_tokens) != (
            expected_scores + 1
        ):
            continue

        cv_fold = int(
            numeric_tokens[0]
        )

        scores = [
            float(value)
            for value in numeric_tokens[1:]
        ]

        score_idx = 0

        for snp_count in snp_counts:

            for model in models:

                rows.append(
                    {
                        "cv_fold": cv_fold,
                        "snp_count": snp_count,
                        "model": model,
                        "score": scores[
                            score_idx
                        ],
                    }
                )

                score_idx += 1

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No rows parsed from Table S{table_num}"
        )

    # Validate each fold
    per_fold_counts = (
        df.groupby("cv_fold")
        .size()
    )

    if not (
        per_fold_counts == expected_scores
    ).all():

        raise ValueError(
            f"Table S{table_num} validation failed: "
            "malformed CV rows"
        )

    output_path = (
        outdir / output_name
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return df


# ============================================================
# OPTIONAL: TABLES S4 / S5
# GWAS significant loci
# ============================================================

def extract_gwas_loci(
    page_texts: list[str],
    starts: dict[int, int],
    table_num: int,
    output_name: str,
    outdir: Path,
) -> pd.DataFrame:

    text = get_table_block(
        page_texts,
        starts,
        table_num,
    )

    row_pattern = re.compile(
        r"^([BC]\d+)\s+"
        r"(\d+)\s+"
        r"(\d+)\s+"
        r"(\d+)\s+"
        r"([^\s]+)\s+"
        r"(\d+)\s+"
        r"([0-9]+(?:\.[0-9]+)?)$"
    )

    rows = []

    for raw_line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            raw_line.strip(),
        )

        match = row_pattern.match(line)

        if not match:
            continue

        (
            region_name,
            chromosome,
            region_start,
            region_end,
            significant_marker,
            marker_number,
            pve,
        ) = match.groups()

        rows.append(
            {
                "region_name": region_name,
                "chromosome": int(chromosome),
                "region_start": int(region_start),
                "region_end": int(region_end),
                "significant_marker": significant_marker,
                "marker_number": int(marker_number),
                "pve": float(pve),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No rows parsed from Table S{table_num}"
        )

    output_path = (
        outdir / output_name
    )

    df.to_csv(
        output_path,
        index=False,
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract selected supplementary tables "
            "into analysis-ready CSV files."
        )
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help=(
            "Path to "
            "'1_Supplementary Table1-13.pdf'"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--include-gwas",
        action="store_true",
        help=(
            "Also extract Tables S4 and S5 "
            "as GWAS loci CSV files."
        ),
    )

    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(
            f"PDF not found: {args.pdf}"
        )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading PDF:\n{args.pdf.resolve()}\n"
    )

    page_texts = extract_page_texts(
        args.pdf
    )

    table_starts = find_table_starts(
        page_texts
    )

    # -----------------------------
    # S1
    # -----------------------------
    s1 = extract_s1(
        page_texts,
        table_starts,
        args.outdir,
    )

    # -----------------------------
    # S3
    # -----------------------------
    s3 = extract_s3(
        page_texts,
        table_starts,
        args.outdir,
    )

    # -----------------------------
    # S12
    # -----------------------------
    s12 = extract_cv_grid(
        page_texts=page_texts,
        starts=table_starts,
        table_num=12,
        snp_counts=[
            "1",
            "10",
            "50",
            "100",
            "500",
            "2000",
            "5000",
            "All",
        ],
        models=[
            "Logistic Regression",
            "Naive Bayes",
            "Random Forest",
            "SVC",
        ],
        output_name=(
            "binary_gs_performance.csv"
        ),
        outdir=args.outdir,
    )

    # -----------------------------
    # S13
    # -----------------------------
    s13 = extract_cv_grid(
        page_texts=page_texts,
        starts=table_starts,
        table_num=13,
        snp_counts=[
            "10",
            "52",
            "100",
            "500",
            "2000",
            "5000",
            "10000",
            "All",
        ],
        models=[
            "Ridge",
            "Lasso",
            "ElasticNet",
            "Random Forest",
            "SVR",
        ],
        output_name=(
            "continuous_gs_performance.csv"
        ),
        outdir=args.outdir,
    )

    print(
        f"✓ S1  -> {len(s1):>4} CNN model rows"
    )

    print(
        f"✓ S3  -> {len(s3):>4} grapevine accessions"
    )

    print(
        f"✓ S12 -> {len(s12):>4} tidy CV rows; "
        f"{s12['cv_fold'].nunique()} folds"
    )

    print(
        f"✓ S13 -> {len(s13):>4} tidy CV rows; "
        f"{s13['cv_fold'].nunique()} folds"
    )

    # -----------------------------
    # Optional S4/S5
    # -----------------------------
    if args.include_gwas:

        s4 = extract_gwas_loci(
            page_texts,
            table_starts,
            4,
            "binary_gwas_loci.csv",
            args.outdir,
        )

        s5 = extract_gwas_loci(
            page_texts,
            table_starts,
            5,
            "continuous_gwas_loci.csv",
            args.outdir,
        )

        print(
            f"✓ S4  -> {len(s4):>4} binary GWAS loci"
        )

        print(
            f"✓ S5  -> {len(s5):>4} continuous GWAS loci"
        )

    print(
        "\nSaved CSV files to:\n"
        f"{args.outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
