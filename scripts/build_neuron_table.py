from pathlib import Path

import numpy as np 
import pandas as pd 

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

ANNOTATIONS_PATH = (
    RAW_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
NEUROTRANSMITTERS_PATH = (
    RAW_DIR / "body-neurotransmitters-male-cns-v1.0.feather"
)
OUTPUT_PATH = PROCESSED_DIR / "canonical_neurons.feather"

ANNOTATION_COLUMNS = [
    "bodyId",
    "instance",
    "type",
    "superclass",
    "class",
    "subclass",
    "somaSide",
    "rootSide",
    "status",
    "statusLabel",
]

NEUROTRANSMITTERS_COLUMNS = [
    "body",
    "consensus_nt",
    "predicted_nt",
    "predicted_nt_confidence",
    "total_nt_predictions",
]

def load_source_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.read_feather(
        ANNOTATIONS_PATH,
        columns=ANNOTATION_COLUMNS
    )

    neurotransmitters = pd.read_feather(
        NEUROTRANSMITTERS_PATH,
        columns=NEUROTRANSMITTERS_COLUMNS
    )

    return annotations, neurotransmitters

def validate_source_keys(
    annotations: pd.DataFrame,
    neurotransmitters: pd.DataFrame,
) -> None:
    if annotations["bodyId"].isna().any():
        raise ValueError("Annotations contain missing bodyId values")

    if neurotransmitters["body"].isna().any():
        raise ValueError("Neurotransmitters contain missing body values")

    annotation_duplicates = annotations["bodyId"].duplicated().sum()
    neurotransmitter_duplicates = neurotransmitters["body"].duplicated().sum()

    if annotation_duplicates:
        raise ValueError(
            f"Annotations contain {annotation_duplicates} duplicate body IDs"
        )

    if neurotransmitter_duplicates:
        raise ValueError(
            "Neurotransmitters contain "
            f"{neurotransmitter_duplicates} duplicate body IDs"
        )

def select_eligible_neurons(
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    eligible = annotations.loc[
        annotations["superclass"].notna()
    ].copy()

    eligible = eligible.rename(columns={"bodyId": "body_id"})
    eligible = eligible.sort_values("body_id")
    eligible = eligible.reset_index(drop=True)

    eligible.insert(
        loc=0,
        column="node_index",
        value=np.arange(len(eligible), dtype=np.int32),
    )

    return eligible

def attach_neurotransmitters(
        neurons: pd.DataFrame,
        neurotransmitters: pd.DataFrame,
) -> pd.DataFrame:
    neurotransmitters = neurotransmitters.rename(
        columns={"body": "body_id"}
    )

    row_count_before = len(neurons)

    joined = neurons.merge(
        neurotransmitters,
        on="body_id",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    if len(joined) != row_count_before:
        raise ValueError(
            "The neurotransmitter join changed the neuron row count"
        )

    return joined 


POLARITY_BY_TRANSMITTER = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}

def add_derived_columns(neurons: pd.DataFrame) -> pd.DataFrame:
    superclass = neurons["superclass"].fillna("")

    neurons["is_sensory"] = superclass.str.contains(
        "sensory",
        regex=False,
    )

    neurons["is_visual_projection"] = superclass.eq(
        "visual_projection"
    )

    neurons["is_intrinsic"] = superclass.str.contains(
        "intrinsic",
        regex=False,
    )

    neurons["is_descending"] = superclass.str.contains(
        "descending",
        regex=False
    )

    neurons["is_efferent"] = superclass.str.contains(
        r"motor|efferent|endocrine",
        regex=True
    )

    neurons["nt_record_available"] = neurons["_merge"].eq("both")

    neurons["consensus_nt"] = neurons["consensus_nt"].fillna("missing")

    neurons["polarity_known"] = neurons["consensus_nt"].isin(
        POLARITY_BY_TRANSMITTER
    )

    neurons["polarity"] = (
        neurons["consensus_nt"]
        .map(POLARITY_BY_TRANSMITTER)
        .fillna(0)
        .astype(np.int8)
    )

    return neurons

# quality report of the neurons table
def print_quality_report(neurons: pd.DataFrame) -> None:
    total = len(neurons)
    missing_nt = (~neurons["nt_record_available"]).sum()
    unknown_polarity = (~neurons["polarity_known"]).sum()

    print("\nCanonical neuron table")
    print(f"  Neurons: {total:,}")
    print(f"  First node index: {neurons['node_index'].min():,}")
    print(f"  Last node index: {neurons['node_index'].max():,}")
    print(f"  Unique body IDs: {neurons['body_id'].nunique():,}")

    print("\nJoin quality")
    print(
        "  Missing neurotransmitter records: "
        f"{missing_nt:,} ({missing_nt / total:.2%})"
    )
    print(
        "  Unknown or non-binary polarity: "
        f"{unknown_polarity:,} ({unknown_polarity / total:.2%})"
    )

    print("\nNeurotransmitter counts")
    print(neurons["consensus_nt"].value_counts().to_string())

    print("\nBiological roles")
    for column in [
        "is_sensory",
        "is_visual_projection",
        "is_intrinsic",
        "is_descending",
        "is_efferent",
    ]:
        print(f"  {column}: {neurons[column].sum():,}")

def save_neuron_table(neurons: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output = neurons.drop(columns=["_merge"])
    output.to_feather(OUTPUT_PATH)

    print(f"\nSaved: {OUTPUT_PATH}")


def main() -> None:
    annotations, neurotransmitters = load_source_tables()

    validate_source_keys(
        annotations,
        neurotransmitters,
    )

    neurons = select_eligible_neurons(annotations)
    neurons = attach_neurotransmitters(
        neurons,
        neurotransmitters,
    )
    neurons = add_derived_columns(neurons)

    print_quality_report(neurons)
    save_neuron_table(neurons)


if __name__ == "__main__":
    main()