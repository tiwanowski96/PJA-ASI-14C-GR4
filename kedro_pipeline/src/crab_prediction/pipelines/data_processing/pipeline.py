from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_Data, prepare_cleaned_data, enrich_rf_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_Data,
                inputs="crabs",
                outputs="cleaned_crabs",
                name="cleaned_crabs_node",
            ),
            node(
                func=prepare_cleaned_data,
                inputs="cleaned_crabs",
                outputs=["prepared_crabs", "scaler"],
                name="prepare_cleaned_crabs_node",
            ),
            node(
                func=enrich_rf_features,
                inputs="prepared_crabs",
                outputs="enriched_rf_input_table",
                name="create_rf_model_input_table_node",
            ),
        ]
    )
