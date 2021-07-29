# Datasets

We supply several example data files.

* `pointcloud/filtered_2386_9702.laz`  
  This is a 50x50m urban point cloud, courtesy of [CycloMedia](https://www.cyclomedia.com/). See notes below for details.
* `ahn/ahn_2386_9702.laz`  
  This is an AHN point cloud of the same area, which can be generated using the [preprocessing tools](../notebooks/1.%20AHN%20preprocessing.ipynb).
* `ahn/ahn_2386_9702.npz`  
  This contains the pre-processed ground and building surfaces from the AHN point cloud.
* `bgt/bgt_roads_2386_9702.csv`  
  This file contains polygon information of the parking spots and road surfaces in the area.
* `bgt/custom_points_2386_9702.csv`  
  This contains <x,y> coordinates of pole-like objects and trees. These were collected manually from different sources, but can also be scraped from PDOK.

These files are sufficient to run the [notebooks](../notebooks). Some additional required data files can be downloaded with provided scripts.


## Some notes on the Datasets

This repository was designed to be used with specific data sources:

* LAS point clouds of urban scenes supplied by [CycloMedia](https://www.cyclomedia.com/).
* AHN3 point clouds downloaded from [ArcGIS](https://www.arcgis.com/apps/Embed/index.html?appid=a3dfa5a818174aa787392e461c80f781)
* BGT data down from [PDOK](https://www.pdok.nl/) or the [Amsterdam API](https://map.data.amsterdam.nl/maps/bgtobjecten?).

The latter two sources are specific to The Netherlands.

We follow the naming conventions used by CycloMedia, which are based on _tile codes_. Each tile covers an area of exactly 50x50m, and is marked by the coordinates of the lower left corner following the Dutch _Rijksdriehoeksstelsel + NAP (EPSG:7415)_.

The tile code is generated as follows:

`tilecode = [X-coordinaat/50]_[Y-coordinaat/50]`

In our example data files, this is 2386_9702 which thus translates to (119300, 485100) meters in RD coordinates or roughly (52.35264, 4.86321) degrees.
