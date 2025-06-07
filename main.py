import yaml
from runners.generate_dataset import generate_dataset
from runners.visualize import visualize_results

if __name__ == "__main__":
    config_path = "configs/burgers_1d.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run simulation and save to .h5
    generate_dataset(config)

    # Generate visualizations (optional)
    if config.get("visualize", True):
        visualize_results(config)
