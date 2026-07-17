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
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ink: #233240;
        --muted: #6b7783;
        --line: #e8ebef;
        --soft: #f7f9f8;
        --leaf: #4f7b5d;
        --grape: #6f5a8f;
        --damage: #b86b4b;
        --gold: #b18a45;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            "Inter", Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    h1 {
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--ink);
        margin-bottom: 0.15rem;
    }

    h2 {
        font-weight: 750;
        letter-spacing: -0.02em;
        color: var(--ink);
        margin-top: 0.4rem;
    }

    h3 {
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--ink);
    }

    p, li {
        color: #3f4b56;
        line-height: 1.6;
    }

    hr {
        margin: 1.6rem 0;
        border-color: var(--line);
    }

    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(
            auto-fit,
            minmax(190px, 1fr)
        );
        gap: 14px;
        margin: 16px 0 22px 0;
    }

    .pipeline-card {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px;
        background: #ffffff;
        min-height: 150px;
        box-shadow:
            0 1px 2px rgba(31, 41, 55, 0.04),
            0 6px 20px rgba(31, 41, 55, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .pipeline-card:hover {
        transform: translateY(-2px);
        box-shadow:
            0 2px 4px rgba(31, 41, 55, 0.06),
            0 12px 28px rgba(31, 41, 55, 0.08);
    }

    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 9px;
        background: #eef3ef;
        color: var(--leaf);
        font-weight: 800;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }

    .pipeline-title {
        font-size: 1rem;
        font-weight: 750;
        color: var(--ink);
        margin-bottom: 7px;
    }

    .pipeline-body {
        font-size: 0.92rem;
        color: #54606b;
        line-height: 1.5;
    }

    .gene-card {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 15px 17px;
        background: #ffffff;
        margin: 9px 0;
        box-shadow: 0 1px 2px rgba(31, 41, 55, 0.03);
    }

    .gene-name {
        font-weight: 800;
        color: var(--grape);
        letter-spacing: -0.01em;
        margin-bottom: 5px;
    }

    .lead-in {
        border-left: 3px solid var(--leaf);
        background: #f3f7f4;
        border-radius: 0 10px 10px 0;
        padding: 12px 18px;
        margin: 2px 0 20px 0;
        color: #3d4a54;
        font-size: 1.0rem;
        line-height: 1.55;
    }

    .takeaway {
        border-left: 3px solid var(--grape);
        background: #f5f2fa;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 14px 0 6px 0;
        color: #3d4a54;
        font-size: 1.0rem;
        line-height: 1.55;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(31, 41, 55, 0.03);
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 750;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
    }

    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--line);
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
        font-size: 0.98rem;
    }

    @media (max-width: 900px) {
        .pipeline-card {
            min-height: auto;
        }
        .block-container {
            padding-left: 0.6rem;
            padding-right: 0.6rem;
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

def lead_in(text: str) -> None:
    """A short orienting sentence at the top of each section."""
    st.markdown(
        f'<div class="lead-in">{text}</div>',
        unsafe_allow_html=True,
    )


def takeaway(text: str) -> None:
    """A highlighted 'what you should walk away with' box."""
    st.markdown(
        f'<div class="takeaway">{text}</div>',
        unsafe_allow_html=True,
    )


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


def render_cascade(
    steps: list[tuple[str, str]],
    active_index: int | None = None,
) -> None:
    """Render the CRK3/ACA12 defense-signaling cascade as connected cards.

    Passing an active_index highlights one step so the reader can walk
    through the pathway one biological event at a time.
    """
    blocks: list[str] = []

    for i, (title, _detail) in enumerate(steps):
        is_active = active_index == i
        border = "#6f5a8f" if is_active else "#e5e9ee"
        bg = "#f3eff8" if is_active else "#ffffff"
        weight = "800" if is_active else "700"

        blocks.append(
            '<div style="flex:1 1 150px;min-width:140px;'
            f'border:1.6px solid {border};background:{bg};'
            'border-radius:12px;padding:12px 14px;">'
            '<div style="font-size:0.70rem;color:#8a94a0;'
            'font-weight:700;letter-spacing:0.04em;'
            f'margin-bottom:6px;">STEP {i + 1}</div>'
            f'<div style="font-weight:{weight};color:#25313c;'
            f'font-size:0.94rem;line-height:1.3;">{title}</div>'
            '</div>'
        )

    arrow = (
        '<div style="align-self:center;color:#b7bec7;'
        'font-size:1.2rem;padding:0 2px;">&#8594;</div>'
    )

    row = arrow.join(blocks)

    html = (
        '<div style="display:flex;flex-wrap:wrap;'
        'align-items:stretch;gap:4px;margin:10px 0 4px 0;">'
        f'{row}</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


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
            '<b>Why highlighted.</b> '
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


def chart_note(
    text: str,
) -> None:
    st.markdown(
        f"""
        <div style="
            color: #4d5964;
            font-size: 0.94rem;
            line-height: 1.45;
            margin-top: -0.35rem;
            margin-bottom: 1.0rem;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


# The central defense-signaling cascade reconstructed in the paper
# (Fig. 4E and the Discussion). Used by the "Finding the Genes" tab.
CASCADE_STEPS: list[tuple[str, str]] = [
    (
        "Herbivore attack",
        "A tobacco cutworm (<i>Spodoptera litura</i>) larva chews the leaf. "
        "Mechanical wounding and chemical cues in the larva's oral secretions "
        "are sensed at the plasma membrane within seconds, before any tissue is "
        "visibly lost.",
    ),
    (
        "Cytosolic Ca²⁺ spike",
        "Wounding depolarizes the membrane and calcium ions (Ca²⁺) flood into the "
        "cytosol. This Ca²⁺ rise is the earliest <b>second messenger</b> of the "
        "defense response, the cell turning a physical wound into a biochemical "
        "signal.",
    ),
    (
        "ACA12 shapes the Ca²⁺ signal",
        "<b>ACA12</b> is a plasma-membrane calcium-transporting ATPase, a Ca²⁺ "
        "pump. It keeps the leaf electrically excitable and sculpts the shape and "
        "duration of the Ca²⁺ spike. Knock out <i>ACA12</i> and this Ca²⁺ signal "
        "collapses, intracellular Ca²⁺ drops, and the plant grows far more "
        "susceptible to <i>Spodoptera</i> feeding.",
    ),
    (
        "Ca²⁺/CaM switches on CRK3",
        "Incoming Ca²⁺ binds calmodulin (CaM). This Ca²⁺/CaM complex flips "
        "<b>CRK3</b>, a CDPK-related protein kinase, from its inactive to its "
        "active form. Jasmonic acid (JA) and abscisic acid (ABA) signaling raise "
        "<i>CRK3</i> transcription during herbivory.",
    ),
    (
        "CRK3 phosphorylates WRKY14",
        "Active CRK3 adds a phosphate group to the transcription factor "
        "<b>WRKY14</b>, switching it on. This is the hand-off from a fast "
        "cytoplasmic signal to a slower change in which genes the nucleus reads.",
    ),
    (
        "Defense output PDF1.2",
        "WRKY14 upregulates <b>PDF1.2</b>, a plant defensin, and the downstream "
        "anti-herbivore program fires. This matches the fact that overexpressing "
        "CRK3 strengthens defense against the cutworm while knocking it out "
        "weakens it.",
    ),
]


# ============================================================
# HEADER
# ============================================================

st.title(
    "🍇 AI Grape Breeder"
)

st.markdown(
    '<div style="color:#6b7783;font-size:1.03rem;margin-top:-0.2rem;">'
    "How deep learning turns photos of chewed grape leaves into pest-resistance "
    "genes and into a way to breed tougher grapevines without spraying more "
    "pesticide."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "🌿 The Challenge",
        "🧪 Plant Biology",
        "🍃 Teaching AI to See",
        "📊 231 Grapevines",
        "🧬 Finding the Genes",
        "🎯 Predicting Resistance",
        "🔭 Future Directions",
    ]
)

(
    tab_overview,
    tab_biology,
    tab_image,
    tab_phenotypes,
    tab_gwas,
    tab_gs,
    tab_future,
) = tabs


# ============================================================
# TAB 1 — THE CHALLENGE (BACKGROUND, RQ, HYPOTHESIS)
# ============================================================

with tab_overview:
    st.header(
        "Breeding a grapevine that fights back"
    )

    st.markdown(
        """
        The tobacco cutworm, *Spodoptera litura*, is one of the most damaging
        pests in grape cultivation. Its larvae chew through grape leaves and
        devour the **mesophyll**, the inner leaf tissue packed with chloroplasts
        where the vine runs almost all of its photosynthesis. What remains is
        perforated, lacy foliage.

        The damage does not stop at the leaf. Outbreaks peak during ripening, so
        heavy feeding disrupts grape coloring and sugar accumulation that season,
        and it drains the stored carbon the vine needs for healthy **bud break
        and flowering the next spring**. A pest problem in August becomes a yield
        problem next year.
        """
    )

    st.markdown(
        """
        The usual answer is to spray pesticides. But pesticides pollute, harm
        nontarget organisms, and drive the rapid evolution of pesticide
        resistance, so the chemicals slowly stop working. A more durable solution
        already lives in the plants themselves, because some grapevine accessions
        are naturally more resistant than others. This natural resistance has
        never been bred systematically into commercial varieties.
        """
    )

    st.markdown(
        '<div class="takeaway"><b>The bottleneck this study attacks.</b> To breed '
        "for resistance we first have to <i>measure</i> it in thousands of plants. "
        "Breeders traditionally score leaf damage by eye into coarse buckets, mild, "
        "moderate, or severe. Eyeballing is slow, subjective, needs years of "
        "training, and discards information, and good genetics cannot come from "
        "coarse hand-scored categories. This paper replaces the human eye with "
        "deep learning.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Grapevine accessions",
            len(phenotypes),
        )

    with c2:
        st.metric(
            "Scored as mild",
            int(
                phenotypes["binary_label"]
                .eq("mild")
                .sum()
            ),
        )

    with c3:
        st.metric(
            "Scored as severe",
            int(
                phenotypes["binary_label"]
                .eq("severe")
                .sum()
            ),
        )

    with c4:
        st.metric(
            "Candidate genes found",
            len(candidate_genes),
        )

    st.divider()

    st.subheader(
        "Research question"
    )

    st.markdown(
        """
        Can image-based deep learning turn visible herbivore damage into
        phenotypes precise enough to (1) capture biologically real differences in
        pest resistance among grapevine accessions, (2) map those differences to
        specific regions of the genome, and (3) predict a vine's resistance from
        its DNA markers before it is ever exposed to the pest?
        """
    )

    st.subheader(
        "Hypothesis"
    )

    st.markdown(
        """
        Genetically diverse grapevine accessions should show **reproducible**
        differences in leaf-damage severity after cutworm feeding. If those
        differences are heritable, then image-derived damage phenotypes, both a
        simple *binary* class (mild or severe) and a fine-grained *continuous*
        score (0–5), will keep enough biological signal to pinpoint
        resistance-associated loci by GWAS and to support marker-based prediction
        of resistance. Crucially, the continuous score should resolve more than the
        binary one, because it can detect small-effect genes that a coarse
        two-category label hides.
        """
    )

    st.divider()

    st.subheader(
        "Roadmap of the study"
    )

    render_pipeline(
        [
            (
                "Plant biology",
                "Why grapevines vary in resistance, and the defenses and signaling "
                "pathways a plant deploys against a chewing insect. (Plant Biology "
                "tab)",
            ),
            (
                "Method",
                "Teach convolutional neural networks to read leaf photos and output "
                "objective damage phenotypes in ~0.03 s per leaf. (Teaching AI to "
                "See tab)",
            ),
            (
                "Result 1, variation",
                "Apply the models to 231 accessions and quantify the natural spread "
                "of resistance across the panel. (231 Grapevines tab)",
            ),
            (
                "Result 2, genes",
                "GWAS links damage phenotypes to 69 QTLs and 139 candidate genes in "
                "real defense pathways. (Finding the Genes tab)",
            ),
            (
                "Result 3, prediction",
                "Genomic selection predicts resistance from DNA markers alone, "
                "so breeders can act without waiting for an outbreak. (Predicting "
                "Resistance tab)",
            ),
        ]
    )

    st.subheader(
        "Why it is important"
    )

    st.markdown(
        """
        Deep learning in agriculture had mostly been used to *identify pests* from
        photos of the insects. This study measures the **plant's response** to the
        pest, objectively and on a continuous scale, then chains that measurement
        to breeding-ready genetic predictions. It turns a slow, subjective human
        task into a fast, scalable phenotyping platform, and shows that a better
        phenotype yields better genetics. The same framework could point at other
        crops and other pests.
        """
    )


# ============================================================
# TAB 2 — PLANT BIOLOGY
# ============================================================

with tab_biology:
    st.header(
        "How a rooted plant defends itself"
    )

    lead_in(
        "Before we measure resistance, we need to know what resistance actually "
        "is at the cellular level."
    )

    st.markdown(
        """
        Plants and insects have shared the planet for over **350 million years**.
        A rooted plant cannot run, hide, or swat, so it defends itself with
        chemistry, structure, and signaling. The pest here is the tobacco cutworm,
        *Spodoptera litura* (Lepidoptera, Noctuidae). Its larvae feed straight on
        grape leaves, so resistance is the outcome of a constant negotiation
        between the caterpillar's feeding and the vine's defenses.
        """
    )

    left, right = st.columns(2)

    with left:
        st.subheader(
            "From a bite to a physiological cost"
        )

        st.markdown(
            """
            Larvae eat **mesophyll tissue**, the palisade and spongy cell layers
            packed with **chloroplasts**, the organelles that fix carbon in
            photosynthesis. Every hole chewed is photosynthetic area removed for
            good.

            Lose enough leaf area and the vine has less carbon for growth, fruit,
            and storage reserves. Feeding during ripening degrades **berry coloring
            and maturation** that season, and drains the reserves the vine needs
            for the **next** season's bud break and flowering. A leaf-eating pest
            becomes a two-year yield problem.
            """
        )

        st.subheader(
            "Direct and indirect defenses"
        )

        st.markdown(
            """
            **Direct defenses** act on the herbivore itself.

            - **structural defenses** (toughened tissue, trichomes) that make
              chewing physically harder,
            - **secondary metabolites** that are toxic or repellent to the insect,
            - **anti-nutritional proteins** (for example protease inhibitors) that
              wreck the caterpillar's digestion and slow its growth,
            - **induced defenses** switched on only after wounding, so the plant
              does not pay the cost until it is attacked.

            **Indirect defenses** recruit outside help. Wounded leaves release
            **volatile organic compounds** that attract the herbivore's natural
            enemies, parasitoid wasps and predators, to the plant.
            """
        )

    with right:
        st.subheader(
            "The signaling network that decides the response"
        )

        st.markdown(
            """
            Wounding is not just physical damage, it launches a wave of chemical
            signaling. A network of **phytohormones** reads the alarm and decides
            which defense genes to switch on. The three central hormones can act
            alone, together, or against one another depending on the attacker.

            **Jasmonic acid (JA)** is the master regulator of anti-herbivore and
            wound-induced defenses.

            **Salicylic acid (SA)** cross-talks with the JA pathway and often
            antagonizes it.

            **Ethylene** tunes stress-responsive gene expression and modulates the
            other pathways.
            """
        )

        st.markdown(
            """
            **Calcium (Ca²⁺) signaling** is the fastest layer of all. Within
            seconds of a bite, cytosolic Ca²⁺ rises. Calcium-binding proteins and
            Ca²⁺-dependent kinases read this spike and pass the wound signal
            onward. Calcium bridges the physical wound and the hormone-driven
            change in gene expression, and it is exactly where this study's
            strongest candidate genes live.
            """
        )

        st.subheader(
            "Why accessions differ genetically"
        )

        st.markdown(
            """
            Herbivore resistance is a **quantitative, polygenic trait**, controlled
            by many genes that each add a small amount. Different accessions carry
            different **alleles** in receptors, signaling kinases, transcriptional
            regulators, and metabolite biosynthesis. Summed together, these small
            differences give genetically distinct vines measurably different leaf
            damage under the same pest pressure. This variation is what the study
            set out to detect and dissect.
            """
        )


# ============================================================
# TAB 3 — TEACHING AI TO SEE DAMAGE (METHODS)
# ============================================================

with tab_image:
    st.header(
        "Teaching AI to see leaf damage"
    )

    lead_in(
        "We cannot map resistance genes until we can measure damage on thousands "
        "of leaves, objectively and fast. Here is how photographs become numbers."
    )

    st.markdown(
        """
        Every downstream result rests on one measurement, how badly is this leaf
        chewed. Scoring damage by eye is slow and subjective, so the authors
        trained **deep convolutional neural networks (DCNNs)**, image models that
        learn visual features layer by layer, to read a leaf photograph and report
        damage in two forms. One is a **binary class** (mild or severe). The other
        is a **continuous pest-damage score, PDS** (0–5). A single leaf is scored
        in about 0.03–0.04 seconds.
        """
    )

    render_pipeline(
        [
            (
                "1 · Photograph the leaf",
                "6–10 leaves per accession are photographed after the cutworm "
                "outbreak with a high-resolution camera.",
            ),
            (
                "2 · Detect the leaf (YOLO)",
                "A YOLO object detector (mAP 0.91) finds the leaf and crops away "
                "background so the model only sees plant tissue.",
            ),
            (
                "3 · Standardize the image",
                "Each crop is resized to 224×224 pixels and pixel values normalized "
                "to 0–1. Rotation, flips, and zoom augment the training set.",
            ),
            (
                "4 · Output a phenotype",
                "One network returns a binary class (mild/severe). A second returns "
                "a continuous 0–5 damage score for genetic analysis.",
            ),
        ]
    )

    st.divider()

    st.subheader(
        "Where the training labels came from"
    )

    lead_in(
        "A model can only learn to reproduce labels a human defined. The phenotype "
        "is anchored to careful human scoring, and to cut bias, every image was "
        "scored by three independent raters."
    )

    st.markdown(
        """
        For the **binary** task, leaves with more than 25% of their area damaged
        were labeled *severe* (1), and 25% or less *mild* (0). For the
        **continuous** task, leaves were scored 1–5 in 20% damage bands (1 for up
        to 20% damaged, adding 1 per additional 20%, up to 5 for over 80%). Try the
        two labeling schemes below to see how raw human judgments become a single
        training target.
        """
    )

    labeling_mode = st.radio(
        "Labeling task",
        options=[
            "Binary label (majority vote)",
            "Continuous PDS label (averaged)",
        ],
        horizontal=True,
        key="labeling_method_mode",
    )

    if labeling_mode == "Binary label (majority vote)":
        st.markdown(
            """
            Three raters each call the leaf **mild** or **severe**. The **majority
            vote** becomes the training label, which averages out one rater's
            idiosyncrasy.
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
                "Training label",
                final_binary_label,
            )

    else:
        st.markdown(
            """
            Three raters each assign a numerical damage score. **Averaging** them
            gives the continuous training target, and the spread between raters
            measures how ambiguous a leaf is.
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
                "Mean PDS (training label)",
                f"{final_pds:.2f}",
            )

        with x2:
            st.metric(
                "Lowest score",
                f"{continuous_scores.min():.1f}",
            )

        with x3:
            st.metric(
                "Rater disagreement",
                f"{score_range:.1f}",
            )

        if score_range >= 1.5:
            st.warning(
                "Large rater disagreement. This leaf sits near a boundary, and "
                "its label carries real uncertainty."
            )

    st.divider()

    st.subheader(
        "Which network reads leaves best?"
    )

    lead_in(
        "Six standard image architectures were trained on the same labeled leaves. "
        "The question is which one most reliably reproduces the human damage call "
        "on leaves it has never seen."
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
    chart_note(
        "Each bar is one CNN architecture, taller is better. <b>How to read it.</b> "
        "The height is the fraction of test leaves where the model's mild/severe "
        "call matched the human label. VGG16 wins. That means the "
        "machine reproduces expert scoring closely enough to replace it, at "
        "thousands of leaves an hour."
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
        "Does a bigger network mean a better one?"
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
        chart_note(
            "Parameters (model size, log scale) on the x-axis, accuracy on the "
            "y-axis. <b>How to read it.</b> If bigger were always better, points "
            "would climb to the right. They do not. The trend is slightly "
            "<i>negative</i>, and VGG16 wins with modest complexity. The question "
            "is not how deep the network is but how faithfully it captures the "
            "visible damage phenotype."
        )

    st.divider()

    cm_col, score_col = st.columns(2)

    with cm_col:
        st.subheader(
            "How VGG16 does on unseen leaves"
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
        chart_note(
            "A <b>confusion matrix</b>. Rows are the true human label, columns the "
            "model's prediction. <b>How to read it.</b> The two dark diagonal cells "
            "(176 and 169) are correct calls. The two pale off-diagonal cells are "
            "mistakes, 7 mild leaves called severe and 10 severe leaves called "
            "mild. Almost everything sits on the diagonal, which is what agreement "
            "between model and human looks like."
        )

        cm_accuracy = (
            np.trace(
                confusion_matrix.to_numpy()
            )
            / confusion_matrix.to_numpy().sum()
        )

        st.metric(
            "Test-set accuracy",
            format_pct(cm_accuracy),
        )

    with score_col:
        st.subheader(
            "The continuous scorer (DCNN-PDS)"
        )

        st.markdown(
            """
            The regression model, **DCNN-PDS**, is built on a VGG16 backbone with
            four added residual blocks, a global average pooling layer, and three
            fully connected layers ending in a single 0–5 output.
            """
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
            On 439 held-out test images, the predicted scores tracked the manual
            scores tightly (r = 0.94, R² = 0.88). The average miss is about
            one-third of a point on the 0–5 scale.
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

    takeaway(
        "<b>Method result.</b> Two trained networks now turn any leaf photo into an "
        "objective binary class (VGG16, 95.3% test accuracy) and a continuous 0–5 "
        "score (DCNN-PDS, r = 0.94). We can finally phenotype resistance at scale. "
        "Next we point them at 231 grapevines."
    )


# ============================================================
# TAB 4 — 231 GRAPEVINES (RESULT 1)
# ============================================================

with tab_phenotypes:
    st.header(
        "How resistance varies across 231 grapevines"
    )

    lead_in(
        "Now the trained models earn their keep. Applied to a diverse panel of "
        "vines, they reveal the natural spread of pest resistance."
    )

    st.markdown(
        """
        An **accession** is a distinct, catalogued grapevine line kept as living
        material for research and breeding. This panel of 231 accessions spans
        three groups with different evolutionary histories, European table grapes
        (domesticated *Vitis vinifera*), East-Asian table grapes (*V. vinifera* ×
        *V. labrusca* hybrids), and wine grapes. That spread is deliberate, since
        it maximizes the genetic variation available to map.

        After a natural cutworm outbreak in the greenhouse, 6–10 leaves per
        accession were photographed and run through the models, giving each vine
        two phenotypes.

        - a **binary phenotype**, mild or severe damage,
        - a **continuous phenotype**, a 0–5 severity score.
        """
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
            "Filter by usage",
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
            "Filter by population background",
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
                "Accessions shown",
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
                "The continuous phenotype"
            )

            fig_hist = px.histogram(
                filtered,
                x="continuous_phenotype",
                nbins=20,
                labels={
                    "continuous_phenotype":
                        "Continuous damage score (0–5)"
                },
                color_discrete_sequence=[GRAPE],
            )

            fig_hist.update_layout(
                yaxis_title="Number of accessions",
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

            chart_note(
                "Each bar counts accessions in a damage-score range, further right "
                "means more tissue lost. <b>How to read it.</b> The distribution is "
                "broad and not bunched at either end, so resistance is a continuum, "
                "not two tidy groups."
            )

        with chart_col2:
            st.subheader(
                "The binary phenotype"
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
                        "Number of accessions",
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

            chart_note(
                "The same vines collapsed into two buckets."
                "132 accessions are severe, 99 mild."
            )

        st.divider()

        st.subheader(
            "But a binary label hides real variation"
        )

        lead_in(
            "Watch what the two-bucket label throws away."
        )

        st.markdown(
            """
            The binary and continuous phenotypes come from **two separate models**,
            so the mild/severe label is not the continuous score cut at a
            threshold. That independence makes the next plot informative. It shows
            how much continuous variation is packed inside each binary class.
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
                    "Continuous damage score (0–5)",
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
        chart_note(
            "Every dot is one accession, and each box summarizes the continuous "
            "scores inside one binary class. <b>How to read it.</b> Notice how tall "
            "each box is and how the two clouds overlap. Two vines both labeled "
            "severe can differ a lot in actual damage. For a polygenic trait built "
            "from many small-effect genes, that hidden within-class spread is "
            "exactly the signal a continuous phenotype keeps and a binary one "
            "erases."
        )

        st.subheader(
            "See the hidden spread inside a single class"
        )

        available_binary_labels = [
            label
            for label in ["mild", "severe"]
            if label in filtered["binary_label"].unique()
        ]

        focus_binary_label = st.radio(
            "Zoom into one binary category",
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
                "Accessions in class",
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
                "Range within one label",
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
                    "Continuous damage score (0–5)"
            },
            color_discrete_sequence=[
                MILD
                if focus_binary_label == "mild"
                else SEVERE
            ],
        )

        fig_hidden.update_layout(
            yaxis_title="Number of accessions",
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
        chart_note(
            "All of these vines carry the <i>same</i> binary label, yet their "
            "continuous scores span the range above. <b>How to read it.</b> If the "
            "mild/severe label were the whole story, this histogram would be a "
            "single spike. It is not. Every bit of width is biological information "
            "the binary label discards, and small-effect resistance genes hide in "
            "exactly this width."
        )

        st.divider()

        st.subheader(
            "Does resistance track a vine's background?"
        )

        group_variable = st.radio(
            "Group the continuous score by",
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
                    "Continuous damage score (0–5)"
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
        chart_note(
            "A violin shows the full distribution of scores per group. Wider means "
            "more vines at that score, and the inner box marks the middle 50%. "
            "<b>How to read it.</b> In the paper, European table grapes were the "
            "most resistant group with the lowest mean and median, while the "
            "East-Asian hybrids took the most severe damage."
        )

        with st.expander(
            "Browse the accession-level data"
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
# TAB 5 — GWAS & GENES (RESULT 2)
# ============================================================

with tab_gwas:
    st.header(
        "From damage phenotypes to defense genes"
    )

    lead_in(
        "We have a number for each vine's resistance and a genome for each vine. "
        "This tab asks the payoff question. Which stretches of DNA explain why "
        "some vines get chewed and others do not?"
    )

    st.markdown(
        """
        The 231 accessions differ in both their damage phenotype and their DNA. A
        **genome-wide association study (GWAS)** scans millions of genetic variants
        and asks, for each one, whether a particular version shows up more often in
        vines with more damage or less. A significant hit marks a **quantitative
        trait locus (QTL)**, a genomic region where variation is linked to the
        trait. Because resistance is polygenic, we expect **many** QTLs scattered
        across the genome, each explaining a slice of the variation.
        """
    )

    g1, g2, g3 = st.columns(3)

    with g1:
        st.metric(
            "QTLs (binary trait)",
            len(binary_gwas),
        )

    with g2:
        st.metric(
            "QTLs (continuous trait)",
            len(continuous_gwas),
        )

    with g3:
        st.metric(
            "Candidate genes",
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
            "Where the signals land"
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
                    "Number of significant QTLs",
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
        chart_note(
            "Counts of significant QTLs per chromosome, split by the phenotype that "
            "found them. <b>How to read it.</b> The signals spread across many "
            "chromosomes. That scatter is the genomic signature of a "
            "<i>polygenic</i> trait. Resistance is built from many loci "
            "genome-wide, just as the biology predicted."
        )

    with chart2:
        st.subheader(
            "How much each region explains"
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
                "pve": "PVE (phenotypic variance explained)",
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
        chart_note(
            "<b>PVE</b> means phenotypic variance explained, the share of trait "
            "variation tied to one region. <b>How to read it.</b> Each dot is a "
            "QTL, and the values are modest, roughly 6–8%. No single locus "
            "dominates, which fits a trait built from many small contributors, and "
            "PVE lets researchers rank which regions to chase first."
        )

    st.markdown(
        """
        A GWAS peak is a signpost, not a mechanism. The real work is reading the
        genes inside each associated region and asking whether their known biology
        makes sense as a herbivore defense. Across both phenotypes the study
        reported **69 QTLs** and **139 non-redundant candidate genes**, and the
        strongest candidates fall into the defense pathways from the *Plant
        Biology* tab, jasmonic acid, salicylic acid, ethylene, and calcium
        signaling.
        """
    )

    takeaway(
        "<b>Why the continuous phenotype mattered here.</b> The binary trait "
        "catches large-effect loci but is blind to subtle ones. The continuous "
        "DCNN-PDS score recovered extra small-effect regions, and its candidate "
        "genes were enriched for <b>calcium-ion binding</b>, pointing straight at "
        "the calcium branch of defense signaling."
    )

    st.divider()

    st.subheader(
        "Candidate Gene Explorer"
    )

    lead_in(
        "Browse the genes the GWAS nominated. Filter by which phenotype found the "
        "gene and by chromosome, then pick a gene to see its role."
    )

    gene_filter_col1, gene_filter_col2 = st.columns(2)

    with gene_filter_col1:
        trait_filter = st.selectbox(
            "Phenotype that found the gene",
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
            f"**Found via** {gene_row['trait_sources']}"
        )

        if bool(
            gene_row["shared_between_traits"]
        ):
            st.success(
                "This gene was nominated by both the binary and the continuous "
                "phenotype, independent evidence that strengthens the case for it."
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
        "A calcium-triggered defense cascade"
    )

    lead_in(
        "The two star genes, ACA12 and CRK3, are not just statistical hits. "
        "Transcriptomics and earlier work let the authors assemble them into a "
        "real signaling pathway. Walk through it one step at a time."
    )

    st.markdown(
        """
        Two of the strongest candidates converge on **calcium signaling**, the fast
        early layer of defense. **ACA12** (a plasma-membrane Ca²⁺ pump, associated
        with the continuous phenotype) and **CRK3** (a calcium/calmodulin-related
        kinase, associated with *both* phenotypes) plug into a cascade that turns a
        caterpillar's bite into switched-on defense genes. Use the selector to
        highlight and read each step.
        """
    )

    cascade_labels = [title for title, _ in CASCADE_STEPS]

    cascade_choice = st.radio(
        "Walk the cascade",
        options=["Show whole pathway"] + cascade_labels,
        horizontal=False,
        key="cascade_step",
    )

    if cascade_choice == "Show whole pathway":
        active_index = None
    else:
        active_index = cascade_labels.index(cascade_choice)

    render_cascade(CASCADE_STEPS, active_index)

    if active_index is None:
        chart_note(
            "The full ACA12 → CRK3 → WRKY14 → PDF1.2 pathway. Select a step above to "
            "highlight it and read the biology. In short, a bite raises calcium, "
            "ACA12 shapes that calcium signal, calcium/calmodulin activates the "
            "kinase CRK3, CRK3 switches on the transcription factor WRKY14, and "
            "WRKY14 turns up the defensin PDF1.2."
        )
    else:
        st.markdown(
            f'<div class="takeaway"><b>Step {active_index + 1}, '
            f'{CASCADE_STEPS[active_index][0]}.</b><br>'
            f'{CASCADE_STEPS[active_index][1]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        The elegance is that GWAS (which genes associate with the trait) and
        transcriptomics (which genes change expression under herbivore attack)
        point at the **same** genes. ACA12 mutants lose intracellular calcium and
        turn hypersensitive to *Spodoptera*. CRK3 overexpression strengthens
        defense while its knockout weakens it. The authors propose that ACA12
        supplies the calcium-rich microenvironment CRK3 needs, a testable
        hypothesis, not a settled fact.
        """
    )

    st.subheader(
        "Other defense genes the study highlighted"
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
            "Number of regions to show",
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
# TAB 6 — GENOMIC SELECTION (RESULT 3)
# ============================================================

with tab_gs:
    st.header(
        "Predicting resistance from DNA alone"
    )

    lead_in(
        "Beyond asking which "
        "single genes matter, we ask whether a machine-learning model, fed a "
        "vine's DNA markers, can predict its resistance, so breeders can screen "
        "seedlings without waiting for a pest outbreak."
    )

    st.markdown(
        """
        **GWAS and genomic selection answer different questions.** GWAS asks *which
        regions* associate with resistance, one variant at a time. **Genomic
        selection (GS)** asks something more practical, *can information spread
        across many markers together predict a plant's phenotype?*
        """
    )

    st.markdown(
        """
        That distinction is everything for a polygenic trait. A vine may carry
        dozens of alleles, each nudging resistance a little. No single marker
        predicts the outcome, but a model that **combines** them can. The payoff is
        real. Screen a seedling's DNA and predict whether it will be resistant
        *before* planting it out to the cutworm, turning years of field trials into
        a lab prediction.
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
            "Final model, optimized logistic regression on 2000 markers, reaches "
            "95.7% accuracy on the held-out test set (predicted versus true "
            "phenotype correlation r = 0.94)."
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
            "Final model, optimized support vector regression (SVR) on 10 000 "
            "markers, reaches r = 0.90 between predicted and true damage scores on "
            "the held-out test set."
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
                "Number of genomic markers used",
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
    chart_note(
        "Each line is one machine-learning model. The x-axis is how many DNA "
        "markers it was given, fewest to all, and the y-axis is prediction quality "
        "in cross-validation. <b>How to read it.</b> Accuracy climbs steeply as the "
        "first informative markers come in, then plateaus, showing that predictive "
        "signal is <i>distributed</i> across many loci, the hallmark of a polygenic "
        "trait. Adding every last marker can even hurt, since most carry noise not "
        "signal, so the best models use a GWAS-filtered subset."
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
            "Markers used",
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
        "What this study cannot yet answer"
    )

    lead_in(
        "I think good science ends with a sharper question. The biggest one here is buried "
        "in the phenotype itself, since 'low damage' can mean two very different "
        "things."
    )

    st.subheader(
        "Resistance versus tolerance"
    )

    st.markdown(
        """
        Visible leaf damage does not separate **resistance** from **tolerance**,
        two biologically distinct strategies.

        - A **resistant** vine cuts the herbivore's feeding, growth, or survival,
          so less tissue is eaten in the first place.
        - A **tolerant** vine may be eaten just as much but shrugs it off, holding
          onto photosynthesis, growth, and reproduction despite the injury.

        Both look like less impact, yet a breeder would treat them very
        differently, and they likely rest on different genes. A leaf photo alone
        cannot tell them apart. A stronger follow-up must measure **both the
        herbivore's performance and the plant's recovery**, not just the hole in
        the leaf.
        """
    )

    st.divider()

    st.subheader(
        "Design the next experiment"
    )

    lead_in(
        "Set the conditions below to frame a testable follow-up question in the "
        "language of the study."
    )

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
                "Apply a controlled number and developmental stage of larvae so "
                "every vine faces the same pressure.",
            ),
            (
                "Track tissue loss",
                "Image consumed leaf area repeatedly through time, not just once at "
                "the end.",
            ),
            (
                "Measure herbivore performance",
                "Record larval mass gain, survival, and development, the signature "
                "of true resistance.",
            ),
            (
                "Measure plant function",
                "Pair damage with photosynthesis, growth, and recovery, the "
                "signature of tolerance.",
            ),
            (
                "Measure defense response",
                "Assay defense genes (CRK3, ACA12, PDF1.2), hormones (JA/SA/"
                "ethylene), and calcium signaling.",
            ),
        ]
    )

    st.subheader(
        "The follow-up experiment"
    )

    st.markdown(
        f"""
        Expose genetically characterized grapevine accessions to a standardized
        number and developmental stage of *S. litura* larvae under **{stress}** in
        a **{environment}**, and measure

        - consumed leaf area through repeated imaging (the phenotype this study
          already automated),
        - larval mass gain and survival (does the plant hurt the insect?),
        - leaf gas exchange and photosynthetic performance (does the plant keep
          functioning?),
        - plant growth and recovery after feeding,
        - expression of the nominated defense genes CRK3, ACA12, and PDF1.2, along
          with their upstream hormone and calcium signals.

        Together these would show whether a low-damage vine is *resisting* the
        cutworm through stronger inducible defense, *tolerating* it through
        maintained physiology, or both, and would test the ACA12–CRK3 calcium
        hypothesis directly, not by association.
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
        ## Sources and data provenance

        This app is an educational companion to one primary research study.
        The supplementary tables and supplementary figures are treated as part
        of that same study, not as separate references.

        ### Annotated sources

        **1. Nanjing Agricultural University The Academy of Science. (2026, February 13).**
        *From leaf images to genomes: Deep learning reshapes pest-resistant breeding.*
        **EurekAlert!**

        Link: https://www.eurekalert.org/news-releases/1116574

        **Source type:** Secondary source / institutional science news release.

        **How I use it:** I use this news article to introduce the research
        problem and explain the broader significance of image-based
        pest-damage phenotyping for genomic analysis and breeding. I rely on
        the primary research paper for detailed methods and numerical results.

        ---

        **2. Gan, Y., Liu, Z., Zhang, F., Xu, Q., Wang, X., Xue, H., et al. (2025).**
        *Deep learning empowers genomic selection of pest-resistant grapevine.*
        **Horticulture Research, 12**(8), uhaf128.

        Link: https://doi.org/10.1093/hr/uhaf128

        **Source type:** Primary research article / peer-reviewed journal
        article, including its official supplementary tables and figures.

        **How I use it:** This is the central scientific source for the app.
        I use it for the research question and hypothesis, the biology of
        *Spodoptera litura* herbivory, image-based phenotyping, CNN model
        evaluation, the 231-accession study design, GWAS results, candidate
        genes, genomic selection, and future directions.

        I also use the article's supplementary materials as data and visual
        evidence from the same primary study:

        - **Figure 1A:** used to explain CNN model comparison for binary
          damage classification.
        - **Figure 1B:** used to reconstruct and explain the VGG16 confusion
          matrix.
        - **Figure 1C:** used to explain the DCNN-PDS continuous scoring
          workflow and model architecture.
        - **Figure 1E:** used to explain the relationship between predicted
          continuous damage scores and manual labels.
        - **Figure 2:** used to interpret pest-damage variation across
          grapevine groups and to frame the 231-accession phenotype analysis.
        - **Figure 3:** used to explain how binary and continuous GWAS connect
          pest-damage phenotypes to genomic regions.
        - **Figure 4:** used to explain candidate defense genes and the
          ACA12–CRK3 calcium-signaling hypothesis.
        - **Figure 5:** used to explain genomic selection and prediction of
          pest-resistance phenotypes from DNA markers.
        - **Supplementary Table S1:** used for CNN model performance.
        - **Supplementary Table S3:** used for 231 accession-level phenotypes.
        - **Supplementary Tables S4–S5:** used for binary and continuous
          GWAS loci.
        - **Supplementary Tables S8–S9:** used for candidate genes.
        - **Supplementary Tables S12–S13:** used for genomic-selection
          cross-validation results.

        ---

        **3. Zhou Lab. (2025).**
        *Pest-Resistance: Deep Learning based Genomic Breeding of
        Pest-Resistant Grapevine* [Computer code repository]. GitHub.

        Link: https://github.com/zhouyflab/Pest-Resistance

        **Source type:** Code repository.

        **How I use it:** I use this repository to better understand the
        computational workflow reported by the authors, including binary
        classification and DCNN-PDS continuous scoring. It helps me explain
        image preprocessing, model architecture, and the connection between
        leaf images and pest-damage phenotypes more accurately. My app does
        not claim to fully reproduce the original model-training pipeline.

        ---

        **4. Mitchell, C., Brennan, R. M., Graham, J., & Karley, A. J. (2016).**
        *Plant defense against herbivorous pests: Exploiting resistance and
        tolerance traits for sustainable crop protection.*
        **Frontiers in Plant Science, 7**, 1132.

        Link: https://doi.org/10.3389/fpls.2016.01132

        **Source type:** Peer-reviewed background source.

        **How I use it:** I use this source to strengthen the plant-biology
        background. It supports my explanation that plants can reduce
        herbivore damage through structural traits, secondary metabolites,
        anti-nutritional compounds, inducible defenses, and indirect
        ecological defenses.

        ---

        **5. Stenberg, J. A., & Muola, A. (2017).**
        *How should plant resistance to herbivores be measured?*
        **Frontiers in Plant Science, 8**, 663.

        Link: https://doi.org/10.3389/fpls.2017.00663

        **Source type:** Peer-reviewed background source.

        **How I use it:** I use this source to interpret visible leaf damage
        as a phenotype. My app treats damage severity as an observable outcome
        of the plant–herbivore interaction, but damage is not identical to a
        molecular defense mechanism. This source helps justify why phenotype
        measurement design matters.

        ---

        **6. Peterson, R. K. D., Varella, A. C., & Higley, L. G. (2017).**
        *Tolerance: The forgotten child of plant resistance.*
        **PeerJ, 5**, e3934.

        Link: https://doi.org/10.7717/peerj.3934

        **Source type:** Peer-reviewed background source.

        **How I use it:** I use this source to develop the future-research
        section. A plant can resist herbivory by reducing feeding or herbivore
        performance, whereas a tolerant plant can maintain or recover growth
        and reproduction after damage. This distinction motivates my proposed
        follow-up experiment.

        ---

        ### Data provenance for this app

        | App data file | Original source within Gan et al. (2025) | How it is used in the app |
        |---|---|---|
        | `cnn_model_performance.csv` | Supplementary Table S1 | Compares CNN model performance for binary image classification. |
        | `grapevine_phenotypes.csv` | Supplementary Table S3 | Provides 231 accession-level binary and continuous pest-damage phenotypes. |
        | `binary_gwas_loci.csv` | Supplementary Table S4 | Shows GWAS loci associated with the binary pest-damage phenotype. |
        | `continuous_gwas_loci.csv` | Supplementary Table S5 | Shows GWAS loci associated with the continuous pest-damage phenotype. |
        | `candidate_gene_summary.csv` | Supplementary Tables S8–S9 | Summarizes nonredundant candidate genes from binary and continuous GWAS results. |
        | `highlighted_genes.csv` | Derived from Gan et al. (2025) candidate-gene interpretation | Highlights biologically important genes such as ACA12 and CRK3 for explanation. |
        | `binary_gs_performance.csv` | Supplementary Table S12 | Visualizes cross-validation results for genomic selection using the binary phenotype. |
        | `continuous_gs_performance.csv` | Supplementary Table S13 | Visualizes cross-validation results for genomic selection using the continuous phenotype. |

        ### Important limitations

        This app analyzes extracted published data from Gan et al. (2025).
        It does not rerun the original deep-learning training, GWAS,
        transcriptomic analysis, or genomic-selection pipeline. It is an
        explanatory, interactive final project built to connect plant herbivory,
        image-based phenotyping, genetic association, candidate genes, and
        breeding prediction.
        """
    )
    