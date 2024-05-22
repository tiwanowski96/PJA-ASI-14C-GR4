from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, create_rf_model, get_cross_validation_metrics, get_model_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["enriched_rf_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=create_rf_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="random_forest",
                name="train_model_node",
            ),
            node(
                func=get_cross_validation_metrics,
                inputs=["random_forest", "X_test", "y_test"],
                outputs=None,
                name="cross_validation_model_node",
            ),
            node(
                func=get_model_metrics,
                inputs=["random_forest", "X_test", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            )
        ]
    )
