from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Grape Breeder",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ink: #25313c;
        --muted: #66737f;
        --line: #e5e9ee;
        --soft: #f7f9f8;
        --leaf: #4f7b5d;
        --grape: #6f5a8f;
        --damage: #b86b4b;
        --gold: #b18a45;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    h1 {
        margin-bottom: 0.2rem;
    }

    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(
            auto-fit,
            minmax(180px, 1fr)
        );
        gap: 12px;
        margin: 14px 0 20px 0;
    }

    .pipeline-card {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px;
        background:
            linear-gradient(
                180deg,
                #ffffff 0%,
                #f9fbfa 100%
            );
        min-height: 150px;
        box-shadow:
            0 5px 18px rgba(31, 41, 55, 0.045);
    }

    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #edf3ee;
        color: var(--leaf);
        font-weight: 800;
        margin-bottom: 10px;
    }

    .pipeline-title {
        font-size: 1rem;
        font-weight: 750;
        color: var(--ink);
        margin-bottom: 7px;
    }

    .pipeline-body {
        font-size: 0.92rem;
        color: #4d5964;
        line-height: 1.42;
    }

    .gene-card {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px 16px;
        background: #ffffff;
        margin: 8px 0;
    }

    .gene-name {
        font-weight: 800;
        color: var(--grape);
        margin-bottom: 4px;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 10px 14px;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 650;
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
    }

    div[data-baseweb="tab-list"] {
        gap: 0.2rem;
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    @media (max-width: 900px) {
        .pipeline-card {
            min-height: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

DATA_FILES = {
    "phenotypes": DATA_DIR / "grapevine_phenotypes.csv",
    "cnn": DATA_DIR / "cnn_model_performance.csv",
    "binary_gs": DATA_DIR / "binary_gs_performance.csv",
    "continuous_gs": DATA_DIR / "continuous_gs_performance.csv",
    "binary_gwas": DATA_DIR / "binary_gwas_loci.csv",
    "continuous_gwas": DATA_DIR / "continuous_gwas_loci.csv",
    "candidate_genes": DATA_DIR / "candidate_gene_summary.csv",
    "highlighted_genes": DATA_DIR / "highlighted_genes.csv",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data() -> dict[str, pd.DataFrame]:
    missing = [
        str(path)
        for path in DATA_FILES.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )

    return {
        name: pd.read_csv(path)
        for name, path in DATA_FILES.items()
    }


def validate_data(
    data: dict[str, pd.DataFrame],
) -> None:
    expected_columns = {
        "phenotypes": {
            "accession_id",
            "usage",
            "population_group",
            "binary_label",
            "binary_code",
            "continuous_phenotype",
        },
        "cnn": {
            "model",
            "accuracy_mean",
            "f1_mean",
            "model_complexity",
            "test_accuracy",
            "test_f1",
        },
        "binary_gs": {
            "cv_fold",
            "snp_count",
            "model",
            "score",
        },
        "continuous_gs": {
            "cv_fold",
            "snp_count",
            "model",
            "score",
        },
        "binary_gwas": {
            "region_name",
            "chromosome",
            "region_start",
            "region_end",
            "significant_marker",
            "marker_number",
            "pve",
        },
        "continuous_gwas": {
            "region_name",
            "chromosome",
            "region_start",
            "region_end",
            "significant_marker",
            "marker_number",
            "pve",
        },
        "candidate_genes": {
            "gene_id",
            "chromosome",
            "trait_sources",
            "qtl_regions",
            "shared_between_traits",
            "source_rows",
            "annotation_raw",
        },
        "highlighted_genes": {
            "gene_id",
            "short_name",
            "biological_role",
            "why_highlighted",
        },
    }

    for name, required in expected_columns.items():
        missing = required - set(data[name].columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

    if len(data["phenotypes"]) != 231:
        raise ValueError(
            "Expected 231 grapevine accessions, "
            f"found {len(data['phenotypes'])}."
        )

    if len(data["cnn"]) != 6:
        raise ValueError(
            "Expected 6 CNN models, "
            f"found {len(data['cnn'])}."
        )


try:
    data = load_data()
    validate_data(data)

except (FileNotFoundError, ValueError) as exc:
    st.error("Data loading or validation failed.")
    st.code(str(exc))
    st.stop()


phenotypes = data["phenotypes"].copy()
cnn = data["cnn"].copy()
binary_gs = data["binary_gs"].copy()
continuous_gs = data["continuous_gs"].copy()
binary_gwas = data["binary_gwas"].copy()
continuous_gwas = data["continuous_gwas"].copy()
candidate_genes = data["candidate_genes"].copy()
highlighted_genes = data["highlighted_genes"].copy()


# ============================================================
# VISUAL CONSTANTS
# ============================================================

LEAF = "#4f7b5d"
GRAPE = "#6f5a8f"
DAMAGE = "#b86b4b"
GOLD = "#b18a45"
BLUE = "#527a95"

MILD = "#6f956f"
SEVERE = "#b86b4b"

GRID = "rgba(37,49,60,0.09)"


# ============================================================
# HELPERS
# ============================================================

def render_pipeline(
    steps: list[tuple[str, str]],
) -> None:
    cards: list[str] = []

    for index, (title, body) in enumerate(
        steps,
        start=1,
    ):
        card_html = (
            '<div class="pipeline-card">'
            f'<div class="step-badge">{index}</div>'
            f'<div class="pipeline-title">{title}</div>'
            f'<div class="pipeline-body">{body}</div>'
            '</div>'
        )
        cards.append(card_html)

    pipeline_html = (
        '<div class="pipeline-grid">'
        + "".join(cards)
        + '</div>'
    )

    st.markdown(
        pipeline_html,
        unsafe_allow_html=True,
    )


def render_gene_card(
    gene_id: str,
    short_name: str,
    biological_role: str,
    why_highlighted: str | None = None,
) -> None:
    html = (
        '<div class="gene-card">'
        f'<div class="gene-name">{short_name} · {gene_id}</div>'
        f'<div>{biological_role}</div>'
    )

    if why_highlighted:
        html += (
            '<div style="margin-top:8px;">'
            '<b>Why highlighted:</b> '
            f'{why_highlighted}'
            '</div>'
        )

    html += '</div>'

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def style_plotly(
    fig,
    height: int = 420,
    show_legend: bool | None = None,
):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#34414c",
        ),
        margin=dict(
            l=35,
            r=25,
            t=15,
            b=35,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        colorway=[
            LEAF,
            GRAPE,
            DAMAGE,
            BLUE,
            GOLD,
            "#7c8b95",
        ],
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_color="#25313c",
        ),
        legend_title_text="",
    )

    # Remove Plotly's internal title key entirely.
    # This avoids the "undefined" title seen in some Streamlit/Plotly combinations.
    if hasattr(fig.layout, "_props"):
        fig.layout._props.pop("title", None)

    if show_legend is not None:
        fig.update_layout(
            showlegend=show_legend
        )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#d9dfe5",
        ticks="outside",
        tickcolor="#d9dfe5",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        showline=False,
    )

    return fig


def aggregate_gs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["snp_count", "model"],
            as_index=False,
        )
        .agg(
            mean_score=("score", "mean"),
            sd_score=("score", "std"),
            n_folds=("score", "size"),
        )
    )

    summary["se_score"] = (
        summary["sd_score"]
        / np.sqrt(summary["n_folds"])
    )

    summary["ci95"] = (
        1.96 * summary["se_score"]
    )

    return summary


def add_region_width(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["region_width_bp"] = (
        result["region_end"]
        - result["region_start"]
        + 1
    )

    return result


def format_pct(
    value: float,
) -> str:
    return f"{value * 100:.1f}%"


def to_bool(
    value,
) -> bool:
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


candidate_genes["shared_between_traits"] = (
    candidate_genes["shared_between_traits"]
    .map(to_bool)
)

binary_gs_summary = aggregate_gs(
    binary_gs
)

continuous_gs_summary = aggregate_gs(
    continuous_gs
)

binary_gwas = add_region_width(
    binary_gwas
)

continuous_gwas = add_region_width(
    continuous_gwas
)

TOTAL_GWAS_REGIONS = (
    len(binary_gwas)
    + len(continuous_gwas)
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title(
        "🍇 AI Grape Breeder"
    )

    st.divider()

    st.metric(
        "Grapevine accessions",
        f"{len(phenotypes):,}",
    )

    st.metric(
        "Candidate genes",
        f"{len(candidate_genes):,}",
    )

    st.metric(
        "GWAS regions",
        f"{TOTAL_GWAS_REGIONS:,}",
    )

    st.divider()

    st.caption(
        "Primary-study data extracted from "
        "Gan et al. (2025) and Supplementary Tables "
        "S1, S3–S5, S8–S9, S12–S13."
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🍇 AI Grape Breeder"
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "🧭 Overview",
        "🌿 Plant Biology",
        "📊 231 Grapevines",
        "🍃 Image Phenotyping",
        "🧬 GWAS & Genes",
        "🎯 Genomic Selection",
        "🔭 Future Research",
    ]
)

(
    tab_overview,
    tab_biology,
    tab_phenotypes,
    tab_image,
    tab_gwas,
    tab_gs,
    tab_future,
) = tabs


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab_overview:
    st.header(
        "Research Overview"
    )

    render_pipeline(
        [
            (
                "Herbivore damage",
                "Tobacco cutworm larvae consume grape leaf mesophyll tissue.",
            ),
            (
                "Leaf-damage phenotype",
                "Images yield binary and continuous measures of visible damage.",
            ),
            (
                "Accession variation",
                "Damage phenotypes are compared across 231 grapevine accessions.",
            ),
            (
                "Genetic association",
                "GWAS links phenotypic variation to genomic regions.",
            ),
            (
                "Genomic selection",
                "Marker-based models predict pest-damage phenotypes for breeding.",
            ),
        ]
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Accessions",
            len(phenotypes),
        )

    with c2:
        st.metric(
            "Mild",
            int(
                phenotypes["binary_label"]
                .eq("mild")
                .sum()
            ),
        )

    with c3:
        st.metric(
            "Severe",
            int(
                phenotypes["binary_label"]
                .eq("severe")
                .sum()
            ),
        )

    with c4:
        st.metric(
            "GWAS regions",
            TOTAL_GWAS_REGIONS,
        )

    st.divider()

    q1, q2 = st.columns(2)

    with q1:
        st.subheader(
            "Research question"
        )

        st.markdown(
            """
            Can image-based measurements of herbivore damage capture
            biological differences in pest resistance among grapevine
            accessions, and can those phenotypes be linked to genomic
            regions associated with resistance?
            """
        )

    with q2:
        st.subheader(
            "Hypothesis"
        )

        st.markdown(
            """
            If grapevine accessions differ genetically in herbivore
            resistance, they should show consistent differences in
            leaf-damage severity. Image-derived phenotypes should
            capture this variation and support identification of
            resistance-associated genomic regions and prediction
            from genetic markers.
            """
        )

    st.divider()

    st.subheader(
        "Biological logic of the study"
    )

    st.markdown(
        """
        Herbivore resistance can be expressed as variation in the
        amount of tissue lost after feeding. In this system,
        *Spodoptera litura* larvae consume grape leaf **mesophyll**,
        producing visible perforation and loss of functional leaf area.

        Differences in damage among grapevine accessions may reflect
        variation in structural defenses, defensive metabolites,
        inducible signaling, or other traits that alter herbivore
        feeding and plant response. The study therefore treats visible
        leaf damage as a **phenotype** and asks whether variation in
        that phenotype is associated with genomic variation.
        """
    )

    st.subheader(
        "Why the findings matter"
    )

    st.markdown(
        """
        The study shows that visible herbivore damage can be converted
        into quantitative phenotypes that preserve meaningful variation
        among grapevine accessions.

        Those phenotypes supported mapping of **69 QTLs** and
        **139 nonredundant candidate genes**, including genes connected
        to hormone and calcium-dependent defense signaling. The continuous
        phenotype also retained variation that can be obscured by a simple
        mild/severe category.

        The significance is that high-throughput phenotyping can
        connect observable plant damage to the genetic architecture of a
        complex resistance trait and support breeding decisions.
        """
    )


# ============================================================
# TAB 2 — PLANT BIOLOGY
# ============================================================

with tab_biology:
    st.header(
        "Plant Biology of Herbivore Resistance"
    )

    st.markdown(
        """
        The focal herbivore is the tobacco cutworm,
        *Spodoptera litura*. Its larvae feed directly on grape leaves,
        so resistance emerges from interactions between herbivore
        feeding behavior and plant defense.
        """
    )

    left, right = st.columns(2)

    with left:
        st.subheader(
            "From feeding to loss of leaf function"
        )

        st.markdown(
            """
            Larvae consume **mesophyll tissue**, which contains many
            chloroplast-rich cells responsible for photosynthetic carbon
            assimilation. Feeding produces holes and loss of functional
            leaf area.

            Extensive damage can reduce tissue available for photosynthesis
            and alter carbon allocation to growth, fruit development, and
            storage. In grapevine, damage during ripening may also affect
            maturation and later nutrient translocation.
            """
        )

        st.subheader(
            "Direct and indirect defenses"
        )

        st.markdown(
            """
            Plants can reduce herbivore damage through several mechanisms:

            - **structural defenses** that make feeding more difficult,
            - **secondary metabolites** that deter or intoxicate herbivores,
            - **anti-nutritional proteins** that reduce herbivore performance,
            - **induced defenses** activated after tissue damage,
            - **volatile compounds** that influence herbivores or their enemies.

            These mechanisms can differ among grapevine genotypes.
            """
        )

    with right:
        st.subheader(
            "Defense signaling after herbivory"
        )

        st.markdown(
            """
            Herbivore feeding causes both tissue injury and biochemical
            signaling. Plant cells respond through interacting pathways.

            **Jasmonic acid (JA)**  
            A major regulator of many wound- and herbivore-induced defenses.

            **Salicylic acid (SA)**  
            Participates in defense signaling and interacts with JA pathways.

            **Ethylene**  
            Modulates stress-responsive gene expression and other hormone pathways.

            **Calcium signaling**  
            Rapid changes in cytosolic Ca²⁺ can transmit information from damaged
            tissue and activate downstream kinases and defense responses.
            """
        )

        st.subheader(
            "Why accessions differ"
        )

        st.markdown(
            """
            Herbivore resistance is a **complex, polygenic trait**.
            Different alleles can alter receptors, signaling proteins,
            transcriptional regulation, metabolite production, and other
            components of defense.

            As a result, genetically distinct grapevine accessions can show
            different levels of leaf damage under herbivore pressure.
            """
        )


# ============================================================
# TAB 3 — 231 GRAPEVINES
# ============================================================

with tab_phenotypes:
    st.header(
        "Phenotypic Variation Across 231 Accessions"
    )

    st.markdown(
        """
        Each **accession** is a distinct grapevine entry maintained as
        plant material for research or breeding. Comparing 231 accessions
        allows the study to examine naturally occurring variation in
        herbivore-damage phenotype across diverse genotypes.

        For each accession, the study summarized multiple leaf images into
        two forms of phenotype:

        - a **binary phenotype**: mild or severe damage,
        - a **continuous phenotype**: a numerical estimate of damage severity.
        """
    )

    st.caption(
        "The study used 6–10 leaf images per accession to derive "
        "accession-level phenotypes."
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        usage_options = sorted(
            phenotypes["usage"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_usage = st.multiselect(
            "Usage",
            options=usage_options,
            default=usage_options,
        )

    with filter_col2:
        population_options = sorted(
            phenotypes["population_group"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_populations = st.multiselect(
            "Population metadata",
            options=population_options,
            default=population_options,
        )

    filtered = phenotypes[
        phenotypes["usage"].isin(selected_usage)
        & phenotypes["population_group"].isin(selected_populations)
    ].copy()

    if filtered.empty:
        st.warning(
            "No accessions match the selected filters."
        )

    else:
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                "Accessions",
                len(filtered),
            )

        with m2:
            st.metric(
                "Mean damage score",
                f"{filtered['continuous_phenotype'].mean():.2f}",
            )

        with m3:
            st.metric(
                "Mild",
                format_pct(
                    filtered["binary_label"]
                    .eq("mild")
                    .mean()
                ),
            )

        with m4:
            st.metric(
                "Severe",
                format_pct(
                    filtered["binary_label"]
                    .eq("severe")
                    .mean()
                ),
            )

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader(
                "Continuous damage scores"
            )

            fig_hist = px.histogram(
                filtered,
                x="continuous_phenotype",
                nbins=20,
                labels={
                    "continuous_phenotype":
                        "Continuous damage score"
                },
                color_discrete_sequence=[GRAPE],
            )

            fig_hist.update_layout(
                yaxis_title="Accessions",
                bargap=0.06,
            )

            style_plotly(
                fig_hist,
                height=390,
                show_legend=False,
            )

            st.plotly_chart(
                fig_hist,
                width="stretch",
            )

        with chart_col2:
            st.subheader(
                "Binary damage phenotype"
            )

            binary_counts = (
                filtered["binary_label"]
                .value_counts()
                .reindex(["mild", "severe"])
                .dropna()
                .rename_axis("binary_label")
                .reset_index(name="count")
            )

            fig_binary = px.bar(
                binary_counts,
                x="binary_label",
                y="count",
                color="binary_label",
                text="count",
                color_discrete_map={
                    "mild": MILD,
                    "severe": SEVERE,
                },
                category_orders={
                    "binary_label": [
                        "mild",
                        "severe",
                    ]
                },
                labels={
                    "binary_label":
                        "Binary phenotype",
                    "count":
                        "Accessions",
                },
            )

            style_plotly(
                fig_binary,
                height=390,
                show_legend=False,
            )

            fig_binary.update_traces(
                textposition="outside",
                cliponaxis=False,
            )

            st.plotly_chart(
                fig_binary,
                width="stretch",
            )

        st.divider()

        st.subheader(
            "Binary labels compress continuous variation"
        )

        st.markdown(
            """
            The binary and continuous phenotypes were generated by
            different prediction tasks. The mild/severe label is therefore
            **not** a direct thresholding of the continuous score.
            """
        )

        fig_box = px.box(
            filtered,
            x="binary_label",
            y="continuous_phenotype",
            color="binary_label",
            points="all",
            hover_data=[
                "accession_id",
                "usage",
                "population_group",
            ],
            color_discrete_map={
                "mild": MILD,
                "severe": SEVERE,
            },
            category_orders={
                "binary_label": [
                    "mild",
                    "severe",
                ]
            },
            labels={
                "binary_label":
                    "Binary phenotype",
                "continuous_phenotype":
                    "Continuous damage score",
            },
        )

        style_plotly(
            fig_box,
            height=440,
            show_legend=False,
        )

        fig_box.update_traces(
            jitter=0.35,
            marker=dict(
                size=5,
                opacity=0.55,
            ),
        )

        st.plotly_chart(
            fig_box,
            width="stretch",
        )

        st.subheader(
            "Explore variation within one binary class"
        )

        available_binary_labels = [
            label
            for label in ["mild", "severe"]
            if label in filtered["binary_label"].unique()
        ]

        focus_binary_label = st.radio(
            "Binary category",
            options=available_binary_labels,
            horizontal=True,
            key="binary_variation_focus",
            format_func=str.title,
        )

        same_label_data = (
            filtered.loc[
                filtered["binary_label"]
                == focus_binary_label
            ]
            .copy()
            .sort_values("continuous_phenotype")
        )

        low_row = same_label_data.iloc[0]
        high_row = same_label_data.iloc[-1]

        continuous_range = (
            high_row["continuous_phenotype"]
            - low_row["continuous_phenotype"]
        )

        e1, e2, e3, e4 = st.columns(4)

        with e1:
            st.metric(
                "Accessions",
                len(same_label_data),
            )

        with e2:
            st.metric(
                "Lowest score",
                f"{low_row['continuous_phenotype']:.2f}",
            )

        with e3:
            st.metric(
                "Highest score",
                f"{high_row['continuous_phenotype']:.2f}",
            )

        with e4:
            st.metric(
                "Range",
                f"{continuous_range:.2f}",
            )

        fig_hidden = px.histogram(
            same_label_data,
            x="continuous_phenotype",
            nbins=15,
            hover_data=[
                "accession_id",
                "usage",
                "population_group",
            ],
            labels={
                "continuous_phenotype":
                    "Continuous damage score"
            },
            color_discrete_sequence=[
                MILD
                if focus_binary_label == "mild"
                else SEVERE
            ],
        )

        fig_hidden.update_layout(
            yaxis_title="Accessions",
            bargap=0.06,
        )

        style_plotly(
            fig_hidden,
            height=390,
            show_legend=False,
        )

        st.plotly_chart(
            fig_hidden,
            width="stretch",
        )

        st.divider()

        st.subheader(
            "Continuous phenotype by metadata"
        )

        group_variable = st.radio(
            "Compare by",
            options=[
                "usage",
                "population_group",
            ],
            horizontal=True,
        )

        fig_group = px.violin(
            filtered,
            x=group_variable,
            y="continuous_phenotype",
            box=True,
            points="all",
            hover_data=["accession_id"],
            color=group_variable,
            labels={
                "continuous_phenotype":
                    "Continuous damage score"
            },
        )

        style_plotly(
            fig_group,
            height=450,
            show_legend=False,
        )

        fig_group.update_traces(
            meanline_visible=True,
            opacity=0.78,
        )

        st.plotly_chart(
            fig_group,
            width="stretch",
        )

        with st.expander(
            "Accession-level data"
        ):
            display_cols = [
                "accession_id",
                "usage",
                "population_group",
                "binary_label",
                "continuous_phenotype",
            ]

            st.dataframe(
                filtered[display_cols]
                .sort_values(
                    "continuous_phenotype",
                    ascending=False,
                ),
                width="stretch",
                hide_index=True,
            )


# ============================================================
# TAB 4 — IMAGE PHENOTYPING
# ============================================================

with tab_image:
    st.header(
        "Image-Based Pest-Damage Phenotyping"
    )

    st.markdown(
        """
        Leaf images were used to generate two measurements of
        visible herbivore damage: a **binary damage class** and a
        **continuous pest-damage score**.
        """
    )

    render_pipeline(
        [
            (
                "Leaf photograph",
                "A damaged grape leaf is photographed.",
            ),
            (
                "Leaf-region detection",
                "The leaf is isolated from background image content.",
            ),
            (
                "Image standardization",
                "The cropped image is resized and normalized for model input.",
            ),
            (
                "Damage phenotype",
                "The image yields a binary class or continuous damage score.",
            ),
        ]
    )

    st.divider()

    st.subheader(
        "How human observations became training labels"
    )

    st.markdown(
        """
        To train the models, visible damage first had to be converted
        into reference labels. Human scoring therefore defines the
        phenotype that the image model is trained to reproduce.
        """
    )

    labeling_mode = st.radio(
        "Labeling task",
        options=[
            "Binary label",
            "Continuous PDS label",
        ],
        horizontal=True,
        key="labeling_method_mode",
    )

    if labeling_mode == "Binary label":
        st.markdown(
            """
            Three raters classify the leaf as **mild** or **severe**.
            The majority vote becomes the training label.
            """
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            binary_rater_1 = st.selectbox(
                "Rater 1",
                ["Mild", "Severe"],
                index=1,
                key="binary_rater_1",
            )

        with r2:
            binary_rater_2 = st.selectbox(
                "Rater 2",
                ["Mild", "Severe"],
                index=1,
                key="binary_rater_2",
            )

        with r3:
            binary_rater_3 = st.selectbox(
                "Rater 3",
                ["Mild", "Severe"],
                index=0,
                key="binary_rater_3",
            )

        votes = [
            binary_rater_1,
            binary_rater_2,
            binary_rater_3,
        ]

        mild_votes = votes.count("Mild")
        severe_votes = votes.count("Severe")

        final_binary_label = (
            "Mild"
            if mild_votes > severe_votes
            else "Severe"
        )

        x1, x2, x3 = st.columns(3)

        with x1:
            st.metric(
                "Mild votes",
                mild_votes,
            )

        with x2:
            st.metric(
                "Severe votes",
                severe_votes,
            )

        with x3:
            st.metric(
                "Final label",
                final_binary_label,
            )

    else:
        st.markdown(
            """
            Three numerical damage scores are averaged to form
            the continuous PDS training target.
            """
        )

        s1, s2, s3 = st.columns(3)

        with s1:
            continuous_rater_1 = st.slider(
                "Rater 1 score",
                1.0,
                5.0,
                2.0,
                0.1,
                key="continuous_rater_1",
            )

        with s2:
            continuous_rater_2 = st.slider(
                "Rater 2 score",
                1.0,
                5.0,
                3.0,
                0.1,
                key="continuous_rater_2",
            )

        with s3:
            continuous_rater_3 = st.slider(
                "Rater 3 score",
                1.0,
                5.0,
                4.0,
                0.1,
                key="continuous_rater_3",
            )

        continuous_scores = np.array(
            [
                continuous_rater_1,
                continuous_rater_2,
                continuous_rater_3,
            ],
            dtype=float,
        )

        final_pds = float(
            continuous_scores.mean()
        )

        score_range = float(
            continuous_scores.max()
            - continuous_scores.min()
        )

        x1, x2, x3 = st.columns(3)

        with x1:
            st.metric(
                "Mean PDS",
                f"{final_pds:.2f}",
            )

        with x2:
            st.metric(
                "Lowest score",
                f"{continuous_scores.min():.1f}",
            )

        with x3:
            st.metric(
                "Rater range",
                f"{score_range:.1f}",
            )

        if score_range >= 1.5:
            st.warning(
                "Large rater disagreement indicates label uncertainty."
            )

    st.divider()

    st.subheader(
        "Binary classification: model comparison"
    )

    metric_choice = st.radio(
        "Metric",
        options=[
            "Test accuracy",
            "Test F1",
            "CV mean accuracy",
        ],
        horizontal=True,
    )

    metric_map = {
        "Test accuracy": "test_accuracy",
        "Test F1": "test_f1",
        "CV mean accuracy": "accuracy_mean",
    }

    metric_col = metric_map[metric_choice]

    model_plot_data = cnn.sort_values(
        metric_col,
        ascending=False,
    )

    fig_models = px.bar(
        model_plot_data,
        x="model",
        y=metric_col,
        color="model",
        text=metric_col,
        labels={
            "model": "CNN architecture",
            metric_col: metric_choice,
        },
    )

    style_plotly(
        fig_models,
        height=420,
        show_legend=False,
    )

    fig_models.update_yaxes(
        range=[
            max(
                0,
                cnn[metric_col].min() - 0.04,
            ),
            min(
                1.0,
                cnn[metric_col].max() + 0.025,
            ),
        ]
    )

    fig_models.update_traces(
        texttemplate="%{y:.3f}",
        textposition="outside",
        cliponaxis=False,
    )

    st.plotly_chart(
        fig_models,
        width="stretch",
    )

    best_row = cnn.loc[
        cnn[metric_col].idxmax()
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Top architecture",
            best_row["model"],
        )

    with c2:
        st.metric(
            metric_choice,
            f"{best_row[metric_col]:.3f}",
        )

    with c3:
        st.metric(
            "Parameters",
            f"{int(best_row['model_complexity']):,}",
        )

    with st.expander(
        "Model complexity versus test accuracy"
    ):
        fig_complexity = px.scatter(
            cnn,
            x="model_complexity",
            y="test_accuracy",
            text="model",
            hover_data=[
                "test_f1",
                "accuracy_mean",
            ],
            log_x=True,
            color="model",
            labels={
                "model_complexity":
                    "Parameters (log scale)",
                "test_accuracy":
                    "Test accuracy",
            },
        )

        style_plotly(
            fig_complexity,
            height=420,
            show_legend=False,
        )

        fig_complexity.update_traces(
            textposition="top center",
            marker=dict(size=11),
        )

        st.plotly_chart(
            fig_complexity,
            width="stretch",
        )

    st.divider()

    cm_col, score_col = st.columns(2)

    with cm_col:
        st.subheader(
            "VGG16 test classification"
        )

        confusion_matrix = pd.DataFrame(
            [
                [176, 7],
                [10, 169],
            ],
            index=[
                "Manual mild",
                "Manual severe",
            ],
            columns=[
                "Predicted mild",
                "Predicted severe",
            ],
        )

        fig_cm = px.imshow(
            confusion_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=[
                [0.0, "#f5f7f5"],
                [1.0, LEAF],
            ],
            labels={
                "x": "Predicted label",
                "y": "Manual label",
                "color": "Leaves",
            },
        )

        fig_cm.update_coloraxes(
            showscale=False
        )

        style_plotly(
            fig_cm,
            height=390,
            show_legend=False,
        )

        st.plotly_chart(
            fig_cm,
            width="stretch",
        )

        cm_accuracy = (
            np.trace(
                confusion_matrix.to_numpy()
            )
            / confusion_matrix.to_numpy().sum()
        )

        st.metric(
            "Reconstructed accuracy",
            format_pct(cm_accuracy),
        )

    with score_col:
        st.subheader(
            "Continuous damage scoring"
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Pearson r",
                "0.94",
            )

        with r2:
            st.metric(
                "R²",
                "0.88",
            )

        with r3:
            st.metric(
                "Test MAE",
                "0.3504",
            )

        st.markdown(
            """
            On 439 independent test images, predicted damage scores
            closely tracked manual labels.
            """
        )

    with st.expander(
        "Full CNN performance table"
    ):
        st.dataframe(
            cnn[
                [
                    "model",
                    "accuracy_mean",
                    "f1_mean",
                    "model_complexity",
                    "test_accuracy",
                    "test_f1",
                ]
            ]
            .sort_values(
                "test_accuracy",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TAB 5 — GWAS & GENES
# ============================================================

with tab_gwas:
    st.header(
        "From Damage Phenotypes to Genomic Regions"
    )

    st.markdown(
        """
        The 231 grapevine accessions differ both in pest-damage
        phenotype and in DNA sequence. A genome-wide association
        study (**GWAS**) tests whether particular genetic variants
        occur more often in accessions with different damage phenotypes.

        A significant association identifies a genomic region where
        variation is statistically linked to the phenotype. These regions
        can be interpreted as **quantitative trait loci (QTLs)**.
        Because pest resistance is polygenic, multiple QTLs can contribute
        to variation in damage severity.
        """
    )

    g1, g2, g3 = st.columns(3)

    with g1:
        st.metric(
            "Binary-trait regions",
            len(binary_gwas),
        )

    with g2:
        st.metric(
            "Continuous-trait regions",
            len(continuous_gwas),
        )

    with g3:
        st.metric(
            "Nonredundant candidate genes",
            len(candidate_genes),
        )

    combined_gwas = pd.concat(
        [
            binary_gwas.assign(
                trait_type="Binary phenotype"
            ),
            continuous_gwas.assign(
                trait_type="Continuous phenotype"
            ),
        ],
        ignore_index=True,
    )

    st.divider()

    chart1, chart2 = st.columns(2)

    with chart1:
        st.subheader(
            "Significant regions by chromosome"
        )

        chr_counts = (
            combined_gwas
            .groupby(
                [
                    "chromosome",
                    "trait_type",
                ],
                as_index=False,
            )
            .size()
            .rename(
                columns={
                    "size":
                        "significant_regions"
                }
            )
        )

        fig_chr = px.bar(
            chr_counts,
            x="chromosome",
            y="significant_regions",
            color="trait_type",
            barmode="group",
            color_discrete_map={
                "Binary phenotype": GRAPE,
                "Continuous phenotype": LEAF,
            },
            labels={
                "chromosome": "Chromosome",
                "significant_regions":
                    "Significant regions",
                "trait_type": "Phenotype",
            },
        )

        style_plotly(
            fig_chr,
            height=420,
        )

        st.plotly_chart(
            fig_chr,
            width="stretch",
        )

    with chart2:
        st.subheader(
            "Phenotypic variance explained"
        )

        fig_pve = px.box(
            combined_gwas,
            x="trait_type",
            y="pve",
            color="trait_type",
            points="all",
            hover_data=[
                "region_name",
                "chromosome",
                "significant_marker",
            ],
            color_discrete_map={
                "Binary phenotype": GRAPE,
                "Continuous phenotype": LEAF,
            },
            labels={
                "trait_type": "GWAS phenotype",
                "pve": "PVE",
            },
        )

        style_plotly(
            fig_pve,
            height=420,
            show_legend=False,
        )

        st.plotly_chart(
            fig_pve,
            width="stretch",
        )

    st.caption(
        "PVE = phenotypic variance explained by an associated region."
    )

    st.markdown(
        """
        A GWAS signal does not identify a resistance mechanism by itself.
        Researchers examine genes located within or near an associated region
        and ask whether their known functions are biologically consistent with
        herbivore defense.

        Across the binary and continuous phenotypes, the study reported
        **69 QTLs** and **139 nonredundant candidate genes**. Candidate genes
        were connected to pathways including jasmonic acid, salicylic acid,
        ethylene, and calcium-dependent signaling.
        """
    )

    st.divider()

    st.subheader(
        "Candidate Gene Explorer"
    )

    gene_filter_col1, gene_filter_col2 = st.columns(2)

    with gene_filter_col1:
        trait_filter = st.selectbox(
            "Phenotype source",
            [
                "All genes",
                "Binary phenotype",
                "Continuous phenotype",
                "Shared between both",
            ],
        )

    with gene_filter_col2:
        chromosome_options = (
            ["All chromosomes"]
            + [
                str(value)
                for value in sorted(
                    candidate_genes["chromosome"]
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
            ]
        )

        chromosome_filter = st.selectbox(
            "Chromosome",
            chromosome_options,
        )

    gene_filtered = candidate_genes.copy()

    if trait_filter == "Binary phenotype":
        gene_filtered = gene_filtered[
            gene_filtered["trait_sources"]
            .str.contains(
                "Binary phenotype",
                na=False,
                regex=False,
            )
        ]

    elif trait_filter == "Continuous phenotype":
        gene_filtered = gene_filtered[
            gene_filtered["trait_sources"]
            .str.contains(
                "Continuous phenotype",
                na=False,
                regex=False,
            )
        ]

    elif trait_filter == "Shared between both":
        gene_filtered = gene_filtered[
            gene_filtered["shared_between_traits"]
        ]

    if chromosome_filter != "All chromosomes":
        gene_filtered = gene_filtered[
            gene_filtered["chromosome"]
            .astype(int)
            .eq(int(chromosome_filter))
        ]

    if gene_filtered.empty:
        st.warning(
            "No candidate genes match the filters."
        )

    else:
        selected_gene = st.selectbox(
            "Candidate gene",
            options=(
                gene_filtered["gene_id"]
                .sort_values()
                .tolist()
            ),
        )

        gene_row = (
            gene_filtered.loc[
                gene_filtered["gene_id"]
                .eq(selected_gene)
            ]
            .iloc[0]
        )

        g1, g2, g3, g4 = st.columns(4)

        with g1:
            st.metric(
                "Matching genes",
                len(gene_filtered),
            )

        with g2:
            st.metric(
                "Gene ID",
                gene_row["gene_id"],
            )

        with g3:
            st.metric(
                "Chromosome",
                int(gene_row["chromosome"]),
            )

        with g4:
            st.metric(
                "QTL region(s)",
                gene_row["qtl_regions"],
            )

        st.markdown(
            f"**Phenotype source:** {gene_row['trait_sources']}"
        )

        if bool(
            gene_row["shared_between_traits"]
        ):
            st.success(
                "Detected in both binary and continuous candidate sets."
            )

        curated_match = highlighted_genes[
            highlighted_genes["gene_id"]
            .eq(selected_gene)
        ]

        if not curated_match.empty:
            curated = curated_match.iloc[0]

            render_gene_card(
                gene_id=selected_gene,
                short_name=curated["short_name"],
                biological_role=curated["biological_role"],
                why_highlighted=curated["why_highlighted"],
            )

        with st.expander(
            "Raw supplementary annotation"
        ):
            st.code(
                str(
                    gene_row["annotation_raw"]
                )
            )

    st.divider()

    st.subheader(
        "A calcium-signaling hypothesis"
    )

    st.markdown(
        """
        Two highlighted candidates suggest a possible role for
        calcium-dependent defense signaling.

        **ACA12** encodes a plasma-membrane calcium-transporting
        ATPase and may influence cellular Ca²⁺ homeostasis.

        **CRK3** is associated with calcium/calmodulin-related kinase
        signaling and was linked to both binary and continuous
        phenotypes in the study.

        The authors propose that calcium dynamics could contribute to
        activation of CRK3 or other defense-related proteins after
        herbivore damage.
        """
    )

    st.subheader(
        "Defense-related genes highlighted in the paper"
    )

    highlight_cols = st.columns(2)

    for idx, (_, row) in enumerate(
        highlighted_genes.iterrows()
    ):
        with highlight_cols[idx % 2]:
            render_gene_card(
                gene_id=row["gene_id"],
                short_name=row["short_name"],
                biological_role=row["biological_role"],
            )

    with st.expander(
        "Strongest extracted GWAS regions"
    ):
        top_n = st.slider(
            "Number of regions",
            min_value=5,
            max_value=25,
            value=10,
            step=1,
        )

        top_loci = (
            combined_gwas
            .sort_values(
                "pve",
                ascending=False,
            )
            .head(top_n)
            [
                [
                    "trait_type",
                    "region_name",
                    "chromosome",
                    "region_start",
                    "region_end",
                    "significant_marker",
                    "marker_number",
                    "pve",
                ]
            ]
        )

        st.dataframe(
            top_loci,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TAB 6 — GENOMIC SELECTION
# ============================================================

with tab_gs:
    st.header(
        "Genomic Selection for Pest Resistance"
    )

    st.markdown(
        """
        Genome-wide SNP markers were used to predict binary or
        continuous pest-damage phenotypes.
        """
    )

    st.subheader(
        "How genomic selection differs from GWAS"
    )

    st.markdown(
        """
        GWAS asks which genomic regions are associated with variation
        in pest damage. **Genomic selection asks ** can information 
        distributed across many genetic markers predict the phenotype of a plant?

        This distinction matters for polygenic resistance. A grapevine
        may carry many alleles, each contributing a small amount to
        defense. Genomic selection combines information across markers.

        The practical goal is to predict which accessions are more likely
        to show favorable resistance phenotypes before every plant is
        evaluated extensively under pest pressure.
        """
    )

    gs_mode = st.radio(
        "Prediction task",
        options=[
            "Binary phenotype",
            "Continuous phenotype",
        ],
        horizontal=True,
    )

    if gs_mode == "Binary phenotype":
        gs_summary = binary_gs_summary.copy()

        snp_order = [
            "1",
            "10",
            "50",
            "100",
            "500",
            "2000",
            "5000",
            "All",
        ]

        y_title = "Mean CV accuracy"

        published_result = (
            "Final optimized logistic regression: "
            "95.7% test accuracy."
        )

    else:
        gs_summary = continuous_gs_summary.copy()

        snp_order = [
            "10",
            "52",
            "100",
            "500",
            "2000",
            "5000",
            "10000",
            "All",
        ]

        y_title = "Mean CV Pearson correlation"

        published_result = (
            "Final optimized SVR: "
            "r = 0.90 on the test set."
        )

    gs_summary["snp_count"] = (
        gs_summary["snp_count"]
        .astype(str)
    )

    gs_summary["snp_count"] = pd.Categorical(
        gs_summary["snp_count"],
        categories=snp_order,
        ordered=True,
    )

    gs_summary = gs_summary.sort_values(
        [
            "snp_count",
            "model",
        ]
    )

    fig_gs = px.line(
        gs_summary,
        x="snp_count",
        y="mean_score",
        color="model",
        markers=True,
        error_y="ci95",
        category_orders={
            "snp_count": snp_order
        },
        labels={
            "snp_count":
                "Number of genomic variants",
            "mean_score":
                y_title,
            "model":
                "Model",
        },
    )

    style_plotly(
        fig_gs,
        height=470,
    )

    fig_gs.update_traces(
        line=dict(width=2.5),
        marker=dict(size=7),
    )

    st.plotly_chart(
        fig_gs,
        width="stretch",
    )

    best_config = (
        gs_summary
        .sort_values(
            "mean_score",
            ascending=False,
        )
        .iloc[0]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Best CV model",
            str(
                best_config["model"]
            ),
        )

    with c2:
        st.metric(
            "Variant count",
            str(
                best_config["snp_count"]
            ),
        )

    with c3:
        st.metric(
            "Mean CV score",
            f"{best_config['mean_score']:.3f}",
        )

    st.info(
        published_result
    )


    with st.expander(
        "Cross-validation summary table"
    ):
        st.dataframe(
            gs_summary[
                [
                    "snp_count",
                    "model",
                    "mean_score",
                    "sd_score",
                    "n_folds",
                    "ci95",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


# ============================================================
# TAB 7 — FUTURE RESEARCH
# ============================================================

with tab_future:
    st.header(
        "Design the Next Experiment"
    )

    st.subheader(
        "A key unresolved biological question"
    )

    st.markdown(
        """
        Visible leaf damage does not fully distinguish
        **resistance** from **tolerance**.

        A resistant plant may reduce larval feeding, growth, or survival,
        resulting in less tissue loss. A tolerant plant may experience
        similar damage but maintain photosynthesis, growth, or reproduction
        more effectively after injury.

        A stronger follow-up experiment should therefore measure both
        herbivore performance and plant recovery.
        """
    )

    st.divider()

    f1, f2, f3 = st.columns(3)

    with f1:
        environment = st.selectbox(
            "Environment",
            [
                "controlled greenhouse",
                "commercial vineyard",
                "multiple geographic regions",
            ],
        )

    with f2:
        stress = st.selectbox(
            "Additional condition",
            [
                "normal watering",
                "drought stress",
                "heat stress",
                "mixed pest and disease pressure",
            ],
        )

    with f3:
        target = st.selectbox(
            "Primary question",
            [
                "stability of pest resistance",
                "genotype-by-environment interaction",
                "early damage progression",
                "transferability of genomic prediction",
            ],
        )

    future_question = (
        f"How does {stress} alter {target} among genetically "
        f"characterized grapevines in a {environment}?"
    )

    st.subheader(
        "Future research question"
    )

    st.success(
        future_question
    )

    st.divider()

    st.subheader(
        "Experimental design"
    )

    render_pipeline(
        [
            (
                "Standardize herbivory",
                "Apply a controlled number and developmental stage of larvae.",
            ),
            (
                "Track tissue loss",
                "Measure consumed leaf area repeatedly through time.",
            ),
            (
                "Measure herbivore performance",
                "Record larval mass gain, survival, or development.",
            ),
            (
                "Measure plant function",
                "Pair damage with photosynthesis, growth, and recovery.",
            ),
            (
                "Measure defense response",
                "Test selected defense genes, hormones, or calcium signaling.",
            ),
        ]
    )

    st.subheader(
        "Follow-up experiment"
    )

    st.markdown(
        f"""
        Expose genetically characterized grapevine accessions to a
        standardized number and developmental stage of *S. litura*
        larvae under **{stress}** in a **{environment}**.

        Measure:

        - consumed leaf area through repeated imaging,
        - larval mass gain and survival,
        - leaf gas exchange or photosynthetic performance,
        - plant growth and recovery after feeding,
        - expression of defense-related genes,
        - selected hormone or calcium-signaling responses.

        This design would test whether low visible damage reflects
        reduced herbivore performance, stronger inducible defense,
        greater tolerance of tissue loss, or a combination of these mechanisms.
        """
    )


# ============================================================
# SOURCES
# ============================================================

st.divider()

with st.expander(
    "Sources and data provenance"
):
    st.markdown(
        """
        **Primary research article**

        Gan, Y., et al. (2025).
        *Deep learning empowers genomic selection of pest-resistant grapevine*.
        **Horticulture Research, 12**, uhaf128.

        https://doi.org/10.1093/hr/uhaf128

        **Study data used**

        - S1: CNN model performance
        - S3: 231 accession-level phenotypes
        - S4–S5: GWAS regions
        - S8–S9: candidate genes
        - S12–S13: genomic-selection cross-validation
        """
    )