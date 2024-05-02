from pydantic import BaseModel
from typing import List, Set

class FuzzyCMeansResponse(BaseModel):
    cluster_centers: List[List[float]]

class KMeansResponse(BaseModel):
    labels: List[int]
    cluster_centers: List[List[float]]

class ConnectedComponentsResponse(BaseModel):
    components: List[Set[int]]

class MinimumSpanningTreeResponse(BaseModel):
    pass

class HierarchicalClusteringResponse(BaseModel):
    labels: List[int]

class HierarchicalClusteringLinkageResponse(BaseModel):
    labels: List[int]

class PeakGroupingResponse(BaseModel):
    labels: List[int]

class DifferentialGroupingResponse(BaseModel):
    labels: List[int]

class GaussianClusteringResponse(BaseModel):
    labels: List[int]

class LancesWilliamsClusteringResponse(BaseModel):
    labels: List[int]

class FastReductiveAgglomerativeClusteringResponse(BaseModel):
    labels: List[int]
