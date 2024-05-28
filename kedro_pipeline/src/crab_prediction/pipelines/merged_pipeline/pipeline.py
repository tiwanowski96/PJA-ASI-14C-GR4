from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_models,split_data, create_model,load_champion_model


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
                inputs=["challenger","champion","datasets_list", "params:models_options"],
                outputs=None,
                name="compare_models",
            )
            ,
            node(
                func=load_champion_model,
                inputs=None,
                outputs='champion',
                name="champion_model",
            )
        ]
    )

