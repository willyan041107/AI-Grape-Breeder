from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/candidate_genes.csv"
)

OUTPUT_PATH = Path(
    "data/processed/candidate_gene_summary.csv"
)


def join_unique(values: pd.Series) -> str:
    """
    Join unique non-null values in stable sorted order.
    """
    cleaned = {
        str(value).strip()
        for value in values.dropna()
        if str(value).strip()
    }

    return " | ".join(sorted(cleaned))


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "gene_id",
        "chromosome",
        "qtl_region",
        "trait_type",
        "annotation_raw",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    summary = (
        df.groupby(
            "gene_id",
            as_index=False,
        )
        .agg(
            chromosome=(
                "chromosome",
                "first",
            ),
            trait_sources=(
                "trait_type",
                join_unique,
            ),
            qtl_regions=(
                "qtl_region",
                join_unique,
            ),
            annotation_raw=(
                "annotation_raw",
                join_unique,
            ),
            source_rows=(
                "gene_id",
                "size",
            ),
        )
    )

    summary["shared_between_traits"] = (
        summary["trait_sources"].str.contains(
            "Binary phenotype",
            regex=False,
        )
        &
        summary["trait_sources"].str.contains(
            "Continuous phenotype",
            regex=False,
        )
    )

    summary = summary[
        [
            "gene_id",
            "chromosome",
            "trait_sources",
            "qtl_regions",
            "shared_between_traits",
            "source_rows",
            "annotation_raw",
        ]
    ].sort_values(
        [
            "chromosome",
            "gene_id",
        ]
    )

    # Critical validation against the paper
    if len(summary) != 139:
        raise ValueError(
            "Expected 139 nonredundant candidate genes, "
            f"found {len(summary)}."
        )

    shared_count = int(
        summary["shared_between_traits"].sum()
    )

    if shared_count != 13:
        raise ValueError(
            "Expected 13 genes shared between binary "
            "and continuous analyses, "
            f"found {shared_count}."
        )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"✓ Unique candidate genes: {len(summary)}"
    )

    print(
        f"✓ Shared between traits: {shared_count}"
    )

    print(
        f"✓ Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
