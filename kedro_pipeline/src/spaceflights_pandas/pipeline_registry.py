"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import spaceflights_pandas.pipelines.data_science as ds
import spaceflights_pandas.pipelines.data_processing as dp
import spaceflights_pandas.pipelines.model_evaluation as me


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = ds.create_pipeline()
    data_science_pipeline = dp.create_pipeline()
    model_evaluation_pipeline = me.create_pipeline()

    #pipelines = find_pipelines()
    #pipelines["__default__"] = sum(pipelines.values())
    return {
        "__default__": data_processing_pipeline + data_science_pipeline + model_evaluation_pipeline,
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "model_evaluation": model_evaluation_pipeline,
    }
