"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import crab_prediction.pipelines.data_science as ds
import crab_prediction.pipelines.data_processing as dp
import crab_prediction.pipelines.auto_ml__pipeline as autoMlPip
import crab_prediction.pipelines.deployment_pipeline as deployment
import crab_prediction.pipelines.challenger_pipeline as challenger


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    auto_ml_pipeline = autoMlPip.create_pipeline()
    deployment_pipeline = deployment.create_pipeline()
    challenger_pipeline = challenger.create_pipeline()

    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    return {
        "__default__": data_processing_pipeline + challenger_pipeline + deployment_pipeline,
        "auto_ml": data_processing_pipeline + auto_ml_pipeline,
        "raw": data_processing_pipeline + data_science_pipeline
    }