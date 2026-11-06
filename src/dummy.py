import jax
jax.random.key(0).block_until_ready()

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.custom.custom_model import CustomModelConfig
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
# from lightevaluation import JaxModel

# Set up evaluation tracking
evaluation_tracker = EvaluationTracker(
    output_dir="eval_results",
    save_details=True
)

# Configure the pipeline
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.CUSTOM,
    max_samples=128,
)

# Configure your custom model
model_config = CustomModelConfig(
    model_name="jax_128_adv_v2",
    model_definition_file_path="./lightevaluation.py"
)

# Create and run the pipeline
pipeline = Pipeline(
    tasks="swag",
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model_config=model_config,
)

pipeline.evaluate()
pipeline.show_results()
pipeline.save_and_push_results()