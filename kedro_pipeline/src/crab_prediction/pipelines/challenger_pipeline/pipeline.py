from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, create_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["enriched_rf_input_table", "params:models_options"],
                outputs="datasets_list",
                name="split_data_node",
            ),
            node(
                func=create_model,
                inputs=["datasets_list", "params:models_options"],
                outputs="challenger",
                name="create_challenger_node",
            )
        ]
    )

