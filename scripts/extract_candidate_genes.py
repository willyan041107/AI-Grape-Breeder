from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber


PDF_PATH = Path(
    "data/raw/1_Supplementary Table1-13.pdf"
)

OUT_DIR = Path("data/processed")


TABLE_TITLE_RE = re.compile(
    r"Table\.?\s*S\.?\s*(\d+)\.?",
    re.IGNORECASE,
)


def extract_page_texts(
    pdf_path: Path,
) -> list[str]:
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
    if table_num not in starts:
        raise ValueError(
            f"Could not locate Table S{table_num}"
        )

    start = starts[table_num]

    later_starts = [
        page
        for number, page in starts.items()
        if page > start
    ]

    end = (
        min(later_starts)
        if later_starts
        else len(page_texts)
    )

    return "\n".join(
        page_texts[start:end]
    )


def parse_candidate_gene_rows(
    text: str,
    trait_type: str,
) -> pd.DataFrame:
    """
    Parse candidate-gene rows from S8 or S9.

    The supplementary PDF contains annotation text with
    variable numbers of words, so this parser preserves the
    full annotation tail rather than pretending every text
    field can be split perfectly.
    """
    rows: list[dict[str, object]] = []

    # Each real row begins:
    # row_number GeneID chromosome QTL_region ...
    row_start = re.compile(
        r"^(\d+)\s+"
        r"(Vitvi\d+)\s+"
        r"(\d+)\s+"
        r"([BC]\d+)\s+"
        r"(.*)$"
    )

    for raw_line in text.splitlines():
        line = re.sub(
            r"\s+",
            " ",
            raw_line.strip(),
        )

        match = row_start.match(line)

        if not match:
            continue

        (
            row_number,
            gene_id,
            chromosome,
            qtl_region,
            annotation,
        ) = match.groups()

        rows.append(
            {
                "row_number": int(row_number),
                "gene_id": gene_id,
                "chromosome": int(chromosome),
                "qtl_region": qtl_region,
                "trait_type": trait_type,
                "annotation_raw": annotation.strip(),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No candidate-gene rows parsed "
            f"for {trait_type}"
        )

    return df


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Missing PDF: {PDF_PATH}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    page_texts = extract_page_texts(
        PDF_PATH
    )

    starts = find_table_starts(
        page_texts
    )

    s8_text = get_table_block(
        page_texts,
        starts,
        8,
    )

    s9_text = get_table_block(
        page_texts,
        starts,
        9,
    )

    binary_genes = parse_candidate_gene_rows(
        s8_text,
        trait_type="Binary phenotype",
    )

    continuous_genes = parse_candidate_gene_rows(
        s9_text,
        trait_type="Continuous phenotype",
    )

    binary_path = (
        OUT_DIR
        / "binary_candidate_genes.csv"
    )

    continuous_path = (
        OUT_DIR
        / "continuous_candidate_genes.csv"
    )

    combined_path = (
        OUT_DIR
        / "candidate_genes.csv"
    )

    binary_genes.to_csv(
        binary_path,
        index=False,
    )

    continuous_genes.to_csv(
        continuous_path,
        index=False,
    )

    combined = pd.concat(
        [
            binary_genes,
            continuous_genes,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    print(
        f"✓ S8 -> {len(binary_genes)} "
        "binary candidate-gene rows"
    )

    print(
        f"✓ S9 -> {len(continuous_genes)} "
        "continuous candidate-gene rows"
    )

    print(
        f"✓ Combined -> {len(combined)} rows"
    )

    print(
        "\nSaved to:"
    )

    print(binary_path)
    print(continuous_path)
    print(combined_path)


if __name__ == "__main__":
    main()
