from typing import NamedTuple, List, Tuple
import numpy as np
import networkx as nx
from scipy.sparse import csr_array
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false

class CSRGraph(NamedTuple):
    n: int
    m: int
    offsets: np.ndarray
    nbrs: np.ndarray
    name: str
    labels: np.ndarray | None

    @staticmethod
    def check(g: "CSRGraph"):
        assert len(g.offsets) == g.n + 1
        assert g.offsets[0] == 0
        assert g.offsets[-1] == len(g.nbrs) == 2 * g.m

        assert np.diff(g.offsets).min() > 0
        assert np.unique(g.nbrs).size == g.n

    @classmethod
    def from_edges(cls, n: int, edges: List[Tuple[int, int]], name: str="", labels: np.ndarray | None=None):
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

        # n_self = int((e[:, 0] == e[:, 1]).sum())
        e = e[e[:, 0] != e[:, 1]]

        lo = np.minimum(e[:, 0], e[:, 1])
        hi = np.maximum(e[:, 0], e[:, 1])
        key = lo * n + hi                      # canonical id per undirected edge
        keep = np.unique(key, return_index=True)[1]
        # n_dup = len(key) - len(keep)
        lo, hi = lo[keep], hi[keep]
        m = len(lo)

        src = np.concatenate([lo, hi])         # each edge, both directions
        dst = np.concatenate([hi, lo])

        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(src, minlength=n), out=offsets[1:])

        order = np.lexsort((dst, src))         # group by src, sort within group
        nbrs = dst[order].astype(np.int32)

        result = cls(n, m, offsets, nbrs, name, labels)
        cls.check(result)
        return result

    @classmethod
    def from_networkx(cls, G, name: str=""):
        nodes = list(G.nodes)
        A: csr_array = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.int8)
        A.sum_duplicates()
        A.sort_indices()
        result = cls(G.number_of_nodes(),
                        G.number_of_edges(),
                        A.indptr.astype(np.int32),
                        A.indices.astype(np.int32),
                        name,
                        np.array(nodes))
        cls.check(result)
        return result

    @classmethod
    def from_edge_list_file(cls, path: str, name: str=""):
        n, edges = read_edge_list(path)
        G = create_networkx_graph(n, edges)
        return cls.from_networkx(G, name=name)

    def save_npz(self, path: str):
        np.savez(path, n=self.n, m=self.m, offsets=self.offsets, nbrs=self.nbrs, name=self.name, labels=self.labels if self.labels is not None else "")
    @staticmethod
    def load_npz(path: str) -> "CSRGraph":
        data = np.load(path, allow_pickle=True)
        return CSRGraph(data["n"], data["m"], data["offsets"], data["nbrs"], data["name"].item(), data["labels"] if "labels" in data else None)


def neighbors(csr_graph: CSRGraph, v: int) -> np.ndarray:
    return csr_graph.nbrs[csr_graph.offsets[v]:csr_graph.offsets[v + 1]]
def degree(csr_graph: CSRGraph, v: int) -> int:
    return csr_graph.offsets[v + 1] - csr_graph.offsets[v]
def degrees(csr_graph: CSRGraph) -> np.ndarray:
    return np.diff(csr_graph.offsets)
def max_degree(csr_graph: CSRGraph) -> int:
    return degrees(csr_graph).max()


# Reads edge list from the Stable_set_data-main dataset
def read_edge_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_nodes, num_edges = map(int, lines[0].split())

    edges = []
    for line in lines[1:]:
        split = tuple(map(int, line.split()))
        if len(split) == 3:
            u, v, w = split
        else:
            u, v = split
        edges.append((u-1, v-1))
    
    return num_nodes, edges

# Creates graph from the edges we read
def create_networkx_graph(num_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(0, num_nodes))
    G.add_edges_from(edges)
    return G