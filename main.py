from src.pipeline.training_pipeline import TrainingPipeline
from src.entity.config_entity import TrainingPipelineConfig
from src.exception import CustomerChurnException

import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()

        training_pipeline = TrainingPipeline(
            training_pipeline_config=training_pipeline_config
        )

        training_pipeline.running_pipeline()

    except Exception as e:
        raise CustomerChurnException(e, sys)