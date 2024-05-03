# Clusterization Lab

This is a FastAPI application that provides endpoints for various clustering methods.

## Requirements

- Python 3.9 or higher
- Docker (optional, for containerization)

## Usage
Docker 
```
docker build -t clusterization-lab .
docker run -p 8000:8000 clusterization-lab
```


This will start the FastAPI application inside a Docker container, and it will be accessible at `http://localhost:8000`.

## Endpoints

The following endpoints are available:

- `/k_means/`: Perform k-means clustering.
- `/fuzzy_c_means/`: Perform fuzzy c-means clustering.
- `/connected_components/`: Find connected components in the graph.
- `/minimum_spanning_tree/`: Find the minimum spanning tree of the graph.
- `/hierarchical_clustering/`: Perform hierarchical clustering.
- `/peak_grouping/`: Perform peak grouping.
- `/differential_grouping/`: Perform differential grouping.
- `/gaussian_clustering/`: Perform Gaussian clustering.
- `/lances_williams_clustering/`: Perform Lances-Williams clustering.
- `/fast_reductive_agglomerative_clustering/`: Perform fast reductive agglomerative clustering.

Each endpoint accepts appropriate parameters and returns the result in JSON format.

