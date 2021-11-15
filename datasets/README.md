# Datasets

We supply two point cloud tiles for demonstration purposes, as well as the necessary data sources to run our code. The [notebooks](../notebooks/) provide information on how to extract data for arbitrary point cloud tiles.

* `pointcloud/filtered_23**_970*.laz`  
  Two 50x50m urban point clouds, courtesy of [CycloMedia](https://www.cyclomedia.com/). See notes below for details.
* `ahn/ahn_23**_970*.laz`  
  The corresponding AHN point clouds, which can also be generated using the [preprocessing tools](../notebooks/1.%20AHN%20preprocessing.ipynb).
* `ahn/ahn_23**_970*.npz`  
  The pre-processed ground and building surfaces from the AHN point clouds.
* `bgt/bgt_buildings_demo.csv`  
  BGT building footprint polygons for the two demo tiles.
* `bgt/bgt_roads_demo.csv`  
  BGT road part polygons of the parking spots and road surfaces in the area.
* `bgt/custom_pole_points_demo.csv`  
  This contains <x,y> coordinates of pole-like objects and trees. These were collected manually from different sources, but can also be scraped from PDOK.
  * `bgt/bgt_street_furniture_points_demo.csv`  
  This contains <x,y> coordinates of street furniture objects scraped from BGT.
    
These files are sufficient to run the [notebooks](../notebooks). Some additional required data files can be downloaded with provided scripts.


## Some notes on the Datasets

This repository was designed to be used with specific data sources:

* LAS point clouds of urban scenes supplied by [CycloMedia](https://www.cyclomedia.com/).
* AHN3 or AHN4 point clouds downloaded from [ArcGIS](https://www.arcgis.com/apps/Embed/index.html?appid=a3dfa5a818174aa787392e461c80f781) or [GeoTiles](https://geotiles.nl).
* BGT data down from [PDOK](https://www.pdok.nl/) or the [Amsterdam API](https://map.data.amsterdam.nl/maps/bgtobjecten?).

The latter two sources are specific to The Netherlands.

We follow the naming conventions used by CycloMedia, which are based on _tile codes_. Each tile covers an area of exactly 50x50m, and is marked by the coordinates of the lower left corner following the Dutch _Rijksdriehoeksstelsel + NAP (EPSG:7415)_.

The tile code is generated as follows:

`tilecode = [X-coordinaat/50]_[Y-coordinaat/50]`

In our example data files, this is 2386_9702 which thus translates to (119300, 485100) meters in RD coordinates or roughly (52.35264, 4.86321) degrees.
