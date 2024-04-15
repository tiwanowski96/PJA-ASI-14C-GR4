"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_cross_validation_metrics, get_model_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
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
