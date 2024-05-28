from kedro.pipeline import Pipeline, node, pipeline

from .nodes import champion_loader,compare_models,split_data_for_autogluon,model_chooser,create_autogluon_model, evaluate_autogluon_model,split_data, create_rf_model, get_cross_validation_metrics, get_model_metrics, create_model


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
            ),
            node(
                func=compare_models,
                inputs=["challenger", "champion", "datasets_list", "params:models_options"],
                outputs="champion_checked",
                name="compare_models",
            )
        ]
    )

