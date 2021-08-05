# Urban PointCloud Processing

This repository contains methods for the **automatic classification and labeling of Urban PointClouds** using data fusion. The methods can serve as inspiration, or can be applied as-is under some specific assumptions:

1. Usage in The Netherlands (The "[Rijksdriehoek coordinate system](https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten)");
2. Point clouds in LAS format and tiled following [specific rules](datasets); and
3. Fusion with [AHN](https://www.ahn.nl/) and [BGT](https://www.geobasisregistraties.nl/basisregistraties/grootschalige-topografie) public data sources.

Example [notebooks](notebooks) are provided to demonstrate the tools.

<figure>
  <img
  src="media/examples/demo.gif"
  alt="Example: automatic labeling of ground and buildings.">
  <figcaption class="figure-caption text-center"><b>Example:</b> automatic labeling of ground and buildings.</figcaption>
</figure>

---

## Project Goal

The goal of this project is to automatically locate and classify various assets such as street lights, traffic signs, and trees in street level point clouds. A typical approach would be to build and train a machine learning classier, but this requires a rich labeled dataset to train on. One of the main challenges in working with 3D point cloud data is that, in contrast to 2D computer vision, _no general-purpose training sets are available_. Moreover, the sparsity and non-uniform density of typical point clouds makes transferring results form one task to another difficult.

However, since we are working with urban street level data, we do have access to a large number of public datasets and registries that we can use to start labeling and create an initial training set. This repository contains several **data fusion** methods that combine public datasets and registries such as elevation data and building footprints to automatically label point clouds.

We also provide some **post-processing** methods that further fine-tune the labels. For example, we use region growing to extend the facade of buildings to include protruding elements such as balconies and canopies that are not included in the building footprint.

For a quick dive into this repository take a look at our [complete solution notebook](notebooks/0.%20Complete%20solution.ipynb).

---

## Folder Structure

 * [`datasets`](./datasets) _Demo dataset to get started_
   * [`ahn`](./datasets/ahn) _AHN data_
   * [`bgt`](./datasets/bgt) _BGT data_
   * [`pointcloud`](./datasets/pointcloud) _Example urban point cloud_
 * [`media`](./media) _Visuals_
   * [`examples`](./media/examples)
 * [`notebooks`](./notebooks) _Jupyter notebook tutorials_
 * [`scripts`](./scripts) _Python scripts_
 * [`src`](./src) _Python source code_
   * [`fusion`](./src/fusion) _Data fusion code_
   * [`preprocessing`](./src/preprocessing) _Pre-processing code_
   * [`region_growing`](./src/region_growing) _Region growing code_
   * [`utils`](./src/utils) _Utility functions_



---

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing.git
    ```

2. Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Check out the [notebooks](notebooks) for a demonstration.

---

## Usage

We provide tutorial [notebooks](notebooks) that demonstrate how the tools can be used.

For visualisation of the resulting labelled point clouds we suggest [CloudCompare](https://www.danielgm.net/cc/). Simply open the labelled .laz in CloudCompare, select the cloud, and set `Colors` to the custom `Scalar Field` named `label`.

---

## Containerization

Run the following commands to build your Docker image, and push it to your container registry. See also the `base_image_tag` variable in the `run_on_azure.py` script.

```
# Login to your container registry
$ docker login {registry-name}
# Or if you are using the Azure CLI
$ az acr login --name {registry-name}

# Build and tag the image
$ docker build -t {registry-name}/point-cloud-processing .

# Push the image to the registry
$ docker push {registry-name}/point-cloud-processing

# Run the run_on_azure.py script to run the AHN batch processor remotely
$ python run_on_azure.py
```

You may have to install .NET on your (Linux) host machine. Follow the instructions provided by the [Microsoft .NET documentation](https://docs.microsoft.com/en-us/dotnet/core/install/linux).

---

## Acknowledgements

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.

We owe special thanks to [Dr. Sander Oude-Elberink](https://research.utwente.nl/en/persons/sander-oude-elberink) for ideas and brainstorming regarding data fusion with AHN and BGT data.
