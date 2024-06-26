from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data_for_autogluon, create_autogluon_model, evaluate_autogluon_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data_for_autogluon,
                inputs=["enriched_rf_input_table", "params:split_options"],
                outputs=["train_set","test_set"],
                name="autogluon_split_data_node",
            ),
            node(
                func=create_autogluon_model,
                inputs=["train_set", "params:predictor_options"],
                outputs="predictor",
                name="autogluon_create_predictor_node",
            ),
            node(
                func=evaluate_autogluon_model,
                inputs=["predictor", "test_set"],
                outputs=None,
                name="evaluate_predictor_node",
            )
        ]
    )
