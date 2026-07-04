from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_PATH = Path("app.py")


# ============================================================
# BLOCK 1
# Interactive labeling methods
# ============================================================

METHODS_BLOCK = r'''
    # ========================================================
    # INTERACTIVE LABELING METHODS
    # ========================================================

    st.divider()

    st.subheader(
        "🧪 How Does a Leaf Become Training Data?"
    )

    st.markdown(
        """
        Before a machine-learning model can learn pest damage,
        researchers first need a **target phenotype** for each
        image.

        In the study, three human raters contributed to the
        labeling process. Binary and continuous tasks convert
        those observations into different kinds of training
        targets.
        """
    )

    labeling_mode = st.radio(
        "Choose a labeling task",
        options=[
            "Binary label",
            "Continuous PDS label",
        ],
        horizontal=True,
        key="labeling_method_mode",
    )

    # --------------------------------------------------------
    # Binary label mode
    # --------------------------------------------------------

    if labeling_mode == "Binary label":

        st.markdown(
            """
            ### Majority-vote classification

            Each rater classifies a leaf as **mild** or
            **severe**. The majority decision becomes the final
            binary training label.
            """
        )

        rater_col1, rater_col2, rater_col3 = st.columns(3)

        with rater_col1:
            binary_rater_1 = st.selectbox(
                "Rater 1",
                options=["Mild", "Severe"],
                index=1,
                key="binary_rater_1",
            )

        with rater_col2:
            binary_rater_2 = st.selectbox(
                "Rater 2",
                options=["Mild", "Severe"],
                index=1,
                key="binary_rater_2",
            )

        with rater_col3:
            binary_rater_3 = st.selectbox(
                "Rater 3",
                options=["Mild", "Severe"],
                index=0,
                key="binary_rater_3",
            )

        binary_votes = [
            binary_rater_1,
            binary_rater_2,
            binary_rater_3,
        ]

        mild_votes = binary_votes.count("Mild")
        severe_votes = binary_votes.count("Severe")

        if mild_votes > severe_votes:
            final_binary_label = "Mild"
        else:
            final_binary_label = "Severe"

        st.markdown(
            "### Final training label"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Mild votes",
                mild_votes,
            )

        with result_col2:
            st.metric(
                "Severe votes",
                severe_votes,
            )

        with result_col3:
            st.metric(
                "Majority label",
                final_binary_label,
            )

        st.success(
            f"The model receives **{final_binary_label}** "
            "as the final binary target for this example."
        )

        st.caption(
            "This interaction demonstrates how multiple human "
            "observations can be converted into one categorical "
            "training target."
        )

    # --------------------------------------------------------
    # Continuous label mode
    # --------------------------------------------------------

    else:

        st.markdown(
            """
            ### Averaged continuous pest-damage score

            For the continuous phenotype, raters assign
            numerical pest-damage scores. Their scores are
            averaged to create a more granular training target.
            """
        )

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col1:
            continuous_rater_1 = st.slider(
                "Rater 1 score",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="continuous_rater_1",
            )

        with score_col2:
            continuous_rater_2 = st.slider(
                "Rater 2 score",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                key="continuous_rater_2",
            )

        with score_col3:
            continuous_rater_3 = st.slider(
                "Rater 3 score",
                min_value=1.0,
                max_value=5.0,
                value=4.0,
                step=0.1,
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

        st.markdown(
            "### Final training target"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Mean PDS label",
                f"{final_pds:.2f}",
            )

        with result_col2:
            st.metric(
                "Lowest rater score",
                f"{continuous_scores.min():.1f}",
            )

        with result_col3:
            st.metric(
                "Rater score range",
                f"{score_range:.1f}",
            )

        st.success(
            f"The model receives **{final_pds:.2f}** "
            "as the continuous pest-damage target."
        )

        if score_range >= 1.5:
            st.warning(
                "The raters disagree substantially in this "
                "example. This illustrates an important "
                "data-science issue: training labels can "
                "contain human measurement uncertainty."
            )

        else:
            st.info(
                "The raters are relatively consistent in this "
                "example, producing a stable average score."
            )

    st.markdown(
        """
        #### Why this matters

        The model does not directly observe an abstract concept
        called “resistance.” It learns patterns associated with
        the **phenotype labels supplied during training**.

        Therefore, biological measurement and human annotation
        are part of the data-generation process, not separate
        from the machine-learning model.
        """
    )

'''


# ============================================================
# BLOCK 2
# Why continuous phenotyping matters
# ============================================================

WHY_EXCITING_BLOCK = r'''
        # ====================================================
        # WHY CONTINUOUS PHENOTYPING MATTERS
        # ====================================================

        st.divider()

        st.subheader(
            "🔍 What Does Binary Classification Hide?"
        )

        st.markdown(
            """
            Binary classification is convenient, but a category
            such as **mild** or **severe** can contain substantial
            continuous variation.

            Use the real accession-level data below to explore
            how much biological detail is compressed when a
            continuous phenotype is reduced to a category.
            """
        )

        available_binary_labels = [
            label
            for label in ["mild", "severe"]
            if label in filtered["binary_label"].unique()
        ]

        if available_binary_labels:

            focus_binary_label = st.radio(
                "Choose one binary category",
                options=available_binary_labels,
                horizontal=True,
                key="binary_variation_focus",
                format_func=lambda x: x.title(),
            )

            same_label_data = (
                filtered.loc[
                    filtered["binary_label"]
                    == focus_binary_label
                ]
                .copy()
                .sort_values(
                    "continuous_phenotype"
                )
            )

            if not same_label_data.empty:

                low_row = same_label_data.iloc[0]
                high_row = same_label_data.iloc[-1]

                continuous_range = (
                    high_row["continuous_phenotype"]
                    - low_row["continuous_phenotype"]
                )

                e1, e2, e3, e4 = st.columns(4)

                with e1:
                    st.metric(
                        "Accessions in category",
                        len(same_label_data),
                    )

                with e2:
                    st.metric(
                        "Lowest continuous score",
                        (
                            f"{low_row['continuous_phenotype']:.2f}"
                        ),
                    )

                with e3:
                    st.metric(
                        "Highest continuous score",
                        (
                            f"{high_row['continuous_phenotype']:.2f}"
                        ),
                    )

                with e4:
                    st.metric(
                        "Within-category range",
                        f"{continuous_range:.2f}",
                    )

                fig_hidden_variation = px.histogram(
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
                            "Continuous pest-damage score"
                    },
                    title=(
                        "Continuous variation among accessions "
                        f"labeled '{focus_binary_label.title()}'"
                    ),
                )

                fig_hidden_variation.update_layout(
                    yaxis_title="Number of accessions"
                )

                st.plotly_chart(
                    fig_hidden_variation,
                    width="stretch",
                )

                st.markdown(
                    "### Same binary label, "
                    "different continuous scores"
                )

                example_col1, example_col2 = st.columns(2)

                with example_col1:
                    st.markdown(
                        f"""
                        **Lower-score accession**

                        - ID: `{low_row["accession_id"]}`
                        - Binary label:
                          **{focus_binary_label.title()}**
                        - Continuous score:
                          **{low_row["continuous_phenotype"]:.2f}**
                        - Usage:
                          {low_row["usage"]}
                        - Population metadata:
                          {low_row["population_group"]}
                        """
                    )

                with example_col2:
                    st.markdown(
                        f"""
                        **Higher-score accession**

                        - ID: `{high_row["accession_id"]}`
                        - Binary label:
                          **{focus_binary_label.title()}**
                        - Continuous score:
                          **{high_row["continuous_phenotype"]:.2f}**
                        - Usage:
                          {high_row["usage"]}
                        - Population metadata:
                          {high_row["population_group"]}
                        """
                    )

                st.success(
                    "These two accessions share the same binary "
                    "category but differ in continuous damage "
                    "severity. The continuous phenotype preserves "
                    "variation that the category alone cannot show."
                )

                st.markdown(
                    """
                    #### Why is this exciting?

                    Pest resistance is a complex biological trait.
                    Small differences among plants may reflect
                    multiple genetic effects rather than one
                    simple resistant-versus-susceptible switch.

                    A more granular phenotype can therefore
                    provide richer information for downstream
                    analyses such as GWAS and genomic prediction.

                    This does not automatically prove that
                    continuous scores are always superior, but it
                    explains why preserving subtle phenotypic
                    variation can be biologically valuable.
                    """
                )

                with st.expander(
                    "Show within-category summary statistics"
                ):

                    within_category_summary = (
                        same_label_data[
                            "continuous_phenotype"
                        ]
                        .describe()
                        .rename("value")
                        .reset_index()
                        .rename(
                            columns={
                                "index": "statistic"
                            }
                        )
                    )

                    st.dataframe(
                        within_category_summary,
                        width="stretch",
                        hide_index=True,
                    )

'''


# ============================================================
# PATCH HELPERS
# ============================================================

def insert_before_anchor(
    text: str,
    anchor: str,
    block: str,
    unique_tag: str,
) -> str:
    """
    Insert block immediately before anchor.

    The function is idempotent:
    if unique_tag already exists, no duplicate is inserted.
    """
    if unique_tag in text:
        print(
            f"SKIP: block already exists -> {unique_tag}"
        )
        return text

    position = text.find(anchor)

    if position == -1:
        raise ValueError(
            "Could not find expected anchor:\n"
            f"{anchor}"
        )

    return (
        text[:position]
        + block
        + text[position:]
    )


def main() -> None:
    """
    Patch app.py with:
    1. Interactive labeling methods
    2. Binary-vs-continuous hidden variation analysis
    """
    if not APP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {APP_PATH.resolve()}"
        )

    original = APP_PATH.read_text(
        encoding="utf-8"
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = Path(
        f"app_before_methods_patch_{timestamp}.py"
    )

    shutil.copy2(
        APP_PATH,
        backup_path,
    )

    # --------------------------------------------------------
    # Anchor 1:
    # Insert labeling methods before CNN model comparison
    # --------------------------------------------------------

    methods_anchor = (
        '    st.subheader("Which CNN performed best?")\n'
    )

    patched = insert_before_anchor(
        text=original,
        anchor=methods_anchor,
        block=METHODS_BLOCK,
        unique_tag="# INTERACTIVE LABELING METHODS",
    )

    # --------------------------------------------------------
    # Anchor 2:
    # Insert hidden-variation section before metadata violin plot
    # --------------------------------------------------------

    why_anchor = (
        '        st.subheader(\n'
        '            "Continuous phenotype by original metadata"\n'
        '        )\n'
    )

    patched = insert_before_anchor(
        text=patched,
        anchor=why_anchor,
        block=WHY_EXCITING_BLOCK,
        unique_tag="# WHY CONTINUOUS PHENOTYPING MATTERS",
    )

    # --------------------------------------------------------
    # Save patched app
    # --------------------------------------------------------

    APP_PATH.write_text(
        patched,
        encoding="utf-8",
    )

    print(
        "✓ app.py patched successfully"
    )

    print(
        f"✓ Backup saved to: {backup_path}"
    )

    print(
        "\nAdded modules:"
    )

    print(
        "  1. Interactive labeling methods"
    )

    print(
        "  2. Why continuous phenotyping matters"
    )


if __name__ == "__main__":
    main()