# Urban PointCloud Processing

This repository contains methods for the (pre-)processing of Urban PointClouds. The tools can serve as inspiration, or can be applied as-is under some specific assumptions:

1. Usage in The Netherlands (The "[Rijksdriehoek coordinate system](https://nl.wikipedia.org/wiki/Rijksdriehoeksco%C3%B6rdinaten)");
2. Point clouds in LAS format and tiled following [specific rules](datasets); and
3. Fusion with [AHN](https://www.ahn.nl/) and [BGT](https://www.geobasisregistraties.nl/basisregistraties/grootschalige-topografie) public data sources.

Example [notebooks](notebooks) are provided to demonstrate the tools.


![](media/examples/demo.gif)

---

## Project Folder Structure

```buildoutcfg
├── datasets
│   ├── ahn
│   └── pointcloud
├── media
│   └── examples
├── notebooks
├── scripts
└── src
    ├── fusion
    ├── preprocessing
    └── utils
```

[//]: # (Generate tree with $ tree -d -I __* --noreport)

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

---

## Acknowledgements

This repository was created by [Amsterdam Intelligence](https://amsterdamintelligence.com/) for the City of Amsterdam.

We owe special thanks to [Dr. Sander Oude-Elberink](https://research.utwente.nl/en/persons/sander-oude-elberink) for ideas and brainstorming regarding data fusion with AHN and BGT data.
