from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_models, load_champion_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_champion_model,
                inputs=None,
                outputs='champion',
                name="champion_model",
            ),
            node(
                func=compare_models,
                inputs=["challenger","champion","datasets_list", "params:model_type"],
                outputs="champion_checked",
                name="compare_models",
            )

        ]
    )

