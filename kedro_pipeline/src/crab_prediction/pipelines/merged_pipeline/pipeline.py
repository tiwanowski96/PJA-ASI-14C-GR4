from kedro.pipeline import Pipeline, node, pipeline

from .nodes import champion_loader,compare_models,split_data_for_autogluon,model_chooser,create_autogluon_model, evaluate_autogluon_model,split_data, create_rf_model, get_cross_validation_metrics, get_model_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_chooser,
                inputs=None,
                outputs="my_bool",
                name="model_chooser",
            ),
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
                outputs='automl_challenger',
                name="evaluate_predictor_node",
            ),
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
                outputs='random_challenger',
                name="evaluate_model_node",
            ),node(
                func=champion_loader,
                inputs='my_bool',
                outputs='my_bools',
                name="champion_loader",
            ),
            node(
                func=compare_models,
                inputs=['my_bools', 'automl_challenger', 'random_challenger'],
                outputs="New_champion",
                name="compare_models",
            )
        ]
    )

