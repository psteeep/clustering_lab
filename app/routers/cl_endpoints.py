from fastapi import APIRouter
import numpy as np
from .cluster_methods import ClusterMethod
from ..schemas.schemas import (
    FuzzyCMeansResponse,
    KMeansResponse,
    ConnectedComponentsResponse,
    MinimumSpanningTreeResponse,
    HierarchicalClusteringResponse,
    HierarchicalClusteringLinkageResponse,
    PeakGroupingResponse,
    DifferentialGroupingResponse,
    GaussianClusteringResponse,
    LancesWilliamsClusteringResponse,
    FastReductiveAgglomerativeClusteringResponse
)
from ..utils.gener_data import generate_random_points

router = APIRouter()

@router.get("/fuzzy_c_means/", response_model=FuzzyCMeansResponse)
def fuzzy_c_means_endpoint(n_clusters: int = 3):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    cluster_centers = ClusterMethod.fuzzy_c_means(points, n_clusters)
    return {"cluster_centers": cluster_centers}

@router.get("/k_means/", response_model=KMeansResponse)
def k_means_endpoint(n_clusters: int = 3):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    labels, cluster_centers = ClusterMethod.k_means_clustering(points, n_clusters)
    return {"labels": labels, "cluster_centers": cluster_centers}

@router.get("/connected_components/", response_model=ConnectedComponentsResponse)
def connected_components_endpoint():
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    components = ClusterMethod.connected_components_clustering(points)
    return {"components": components}

@router.get("/hierarchical_clustering", response_model=HierarchicalClusteringResponse)
def hierarchical_clustering_endpoint(n_clusters: int = 3):
    n_points = np.random.randint(50,101)
    points = generate_random_points(n_points)
    labels = ClusterMethod.hierarchical_clustering(points, n_clusters)
    return {"labels": labels}

@router.get("/hierarchical_clustering_linkage/", response_model=HierarchicalClusteringLinkageResponse)
def hierarchical_clustering_linkage_endpoint(n_clusters: int = 3):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    labels = ClusterMethod.hierarchical_clustering_linkage(points, n_clusters)
    return {"labels": labels}


@router.get("/peak_grouping/", response_model=PeakGroupingResponse)
def peak_grouping_endpoint(threshold: float = 0.5, n_clusters: int = None):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    labels = ClusterMethod.peak_grouping(points, threshold, n_clusters)
    return {"labels": labels}

@router.get("/differential_grouping/", response_model=DifferentialGroupingResponse)
def differential_grouping_endpoint(eps: float = 0.5, min_samples: int = 5):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    labels = ClusterMethod.differential_grouping(points, eps, min_samples)
    return {"labels": labels}

@router.get("/gaussian_clustering/", response_model=GaussianClusteringResponse)
def gaussian_clustering_endpoint(n_clusters: int):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    labels = ClusterMethod.gaussian_clustering(points, n_clusters)
    return {"labels": labels}

@router.get("/lances_williams_clustering/", response_model=LancesWilliamsClusteringResponse)
def lances_williams_clustering_endpoint(n_clusters: int):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    labels = ClusterMethod.lances_williams_clustering(points, n_clusters)
    return {"labels": labels}

@router.get("/fast_reductive_agglomerative_clustering/", response_model=FastReductiveAgglomerativeClusteringResponse)
def fast_reductive_agglomerative_clustering_endpoint(n_clusters: int):
    n_points = np.random.randint(50, 101)
    points = generate_random_points(n_points)
    labels = ClusterMethod.fast_reductive_agglomerative_clustering(points, n_clusters)
    return {"labels": labels}