from typing import List, Set, Tuple
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import minimum_spanning_tree

class ClusterMethod:
    @staticmethod
    def fuzzy_c_means(points: List[List[float]], n_clusters: int, fuzziness: int = 2) -> List[List[float]]:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(points)
        return kmeans.cluster_centers_

    @staticmethod
    def k_means_clustering(points: List[List[float]], n_clusters: int) -> Tuple[List[int], List[List[float]]]:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(points)
        return kmeans.labels_, kmeans.cluster_centers_

    @staticmethod
    def connected_components_clustering(points: List[List[float]], threshold: float = 0.5) -> List[Set[int]]:
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i != j and np.linalg.norm(p1 - p2) < threshold:
                    G.add_edge(i, j)
        return list(nx.connected_components(G))

    # @staticmethod
    # def minimum_spanning_tree_clustering(points: List[List[float]]) -> any:
    #     distances = distance.squareform(distance.pdist(points))
    #     mst = minimum_spanning_tree(distances)
    #     return mst

    @staticmethod
    def hierarchical_clustering(points: List[List[float]], n_clusters: int) -> List[int]:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clustering.fit(points)
        return clustering.labels_

    @staticmethod
    def hierarchical_clustering_linkage(points: List[List[float]], n_clusters: int) -> List[int]:
        Z = linkage(points, method='ward', metric='euclidean')
        return fcluster(Z, n_clusters, criterion='maxclust') - 1

    @staticmethod
    def peak_grouping(points: List[List[float]], threshold: float = 0.5, n_clusters: int = None) -> List[int]:
        brc = Birch(threshold=threshold, n_clusters=n_clusters)
        brc.fit(points)
        return brc.labels_

    @staticmethod
    def differential_grouping(points: List[List[float]], eps: float = 0.5, min_samples: int = 5) -> List[int]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(points)
        return db.labels_

    @staticmethod
    def gaussian_clustering(points: List[List[float]], n_clusters: int) -> List[int]:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        gmm.fit(points)
        return gmm.predict(points)

    @staticmethod
    def lances_williams_clustering(points: List[List[float]], n_clusters: int) -> List[int]:
        Z = linkage(points, method='average', metric='euclidean')
        return fcluster(Z, n_clusters, criterion='maxclust') - 1

    @staticmethod
    def fast_reductive_agglomerative_clustering(points: List[List[float]], n_clusters: int) -> List[int]:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        clustering.fit(points)
        return clustering.labels_
