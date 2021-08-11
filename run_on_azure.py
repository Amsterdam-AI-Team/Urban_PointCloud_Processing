import argparse

from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AHN LAS point cloud batch processing on Azure Machine Learning"
    )
    parser.add_argument("--input_dataset_path", type=str, required=True)
    parser.add_argument("--output_dataset_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--compute_target", type=str, default="point-cloud-cpu")
    parser.add_argument("--environment", type=str, default="point-cloud")

    args = parser.parse_args()

    # Set up workspace
    ws = Workspace.from_config(".azure/config.json")
    target = ws.compute_targets[args.compute_target]

    # Set up container environment
    env = Environment(args.environment)
    env.docker.base_image = args.image
    env.python.user_managed_dependencies = True

    exp = Experiment(ws, args.environment)

    # Set up script run configuration
    config = ScriptRunConfig(
        source_directory=".",
        script="scripts/ahn_batch_processor_azure.py",
        arguments=[
            "--input_dataset_path", args.input_dataset_path,
            "--output_dataset_path", args.output_dataset_path,
            "--subscription_id", ws.subscription_id,
            "--resource_group", ws.resource_group,
            "--workspace_name", ws.name,
        ],
        compute_target=target,
        environment=env,
    )

    # submit script to AML
    run = exp.submit(config)
    print(run.get_portal_url()) # link to ml.azure.com
    run.wait_for_completion(show_output=True)
