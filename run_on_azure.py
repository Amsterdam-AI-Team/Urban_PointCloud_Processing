from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
from azureml.data import OutputFileDatasetConfig

base_image_tag = "pointcloudacr.azurecr.io/point-cloud-processing"
input_dataset_name = "point-cloud-input"
input_dataset_path = "/UI/08-04-2021_095741_UTC"

# get workspace
ws = Workspace.from_config(".azure/config.json")

# get compute target
target = ws.compute_targets["point-cloud-cpu"]

# get registered environment
env = Environment("point-cloud")
env.python.user_managed_dependencies = True
env.docker.base_image = base_image_tag

# get/create experiment
exp = Experiment(ws, "point-cloud")

# get blob store mount and datasets
# def_blob_store = ws.get_default_datastore()
# input_dataset = Dataset.File.from_files((def_blob_store, input_dataset_path))
# output = OutputFileDatasetConfig(destination=(def_blob_store, "sample/output")).as_mount().register_on_complete(name="point-cloud-output")
# mount_context = input_dataset.mount()
# mount_context.start()

# set up script run configuration
config = ScriptRunConfig(
    source_directory=".",
    script="scripts/ahn_batch_processor_azure.py",
    compute_target=target,
    environment=env,
)

# submit script to AML
run = exp.submit(config)
print(run.get_portal_url()) # link to ml.azure.com
run.wait_for_completion(show_output=True)
