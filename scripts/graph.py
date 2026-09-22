"""Compressed sparse row (CSR) representation of undirected graphs.
This module provides :class:`CSRGraph`, the immutable graph container used
throughout the package. It also provides helper functions for looking up
neighbours and degrees, and readers for the edge-list files.

CSR layout
----------
A graph with ``n`` vertices (numbered ``0 .. n-1``) and ``m`` undirected
edges is stored in two flat arrays:
 
* ``offsets`` (length ``n + 1``): the neighbour list of ``v`` starts at
  position ``offsets[v]`` in ``nbrs`` and ends before ``offsets[v + 1]``.
  ``offsets[0] == 0`` and ``offsets[n] == 2 * m``.
* ``nbrs`` (length ``2 * m``): all neighbour lists joined end to end. Each
  undirected edge ``{u, v}`` is stored twice: as ``v`` in the list of ``u``,
  and as ``u`` in the list of ``v``.
 
So the neighbours of ``v`` are ``nbrs[offsets[v]:offsets[v + 1]]``, and the
degree of ``v`` is ``offsets[v + 1] - offsets[v]``. The graphs built by this
module keep each neighbour list sorted in increasing order.
 
Example
-------
The path ``0 - 1 - 2`` is stored as::
 
    offsets = [0, 1, 3, 4]
    nbrs    = [1, 0, 2, 1]
"""

from typing import NamedTuple, List, Tuple
import numpy as np
import networkx as nx
from scipy.sparse import csr_array

class CSRGraph(NamedTuple):
    """
    Undirected simple graph in CSR format.

    Attributes
    ----------
    n : int
        Number of vertices. Vertices are the integers ``0 .. n-1``.
    m : int
        Number of undirected edges, each counted once.
    offsets : np.ndarray of int32, shape (n + 1,)
        Start and end positions of each vertex's neighbour list in ``nbrs``.
    nbrs : np.ndarray of int32, shape (2 * m,)
        All neighbour lists joined end to end, each list sorted.
    name : str
        Instance name, used in experiment reports.
    labels : np.ndarray or None
        Optional original vertex labels: ``labels[i]`` is the original label
        of internal vertex ``i``. :meth:`from_networkx` sets this, because
        NetworkX nodes can be any hashable object. It is ``None`` when the
        graph was built from integer edges with no labels given.
    """
    n: int
    m: int
    offsets: np.ndarray
    nbrs: np.ndarray
    name: str
    labels: np.ndarray | None

    @staticmethod
    def check(g: "CSRGraph"):
        """Validate the structural invariants of a CSR graph.
 
        The following conditions are checked:
 
        * ``offsets`` has length ``n + 1`` and ``offsets[0] == 0``;
        * ``offsets[-1] == len(nbrs) == 2 * m``;
        * every vertex has degree at least 1;
        * ``nbrs`` contains exactly ``n`` distinct values, so every vertex
          is some vertex's neighbour."""
        assert len(g.offsets) == g.n + 1
        assert g.offsets[0] == 0
        assert g.offsets[-1] == len(g.nbrs) == 2 * g.m

        assert np.diff(g.offsets).min() > 0
        assert np.unique(g.nbrs).size == g.n

    @classmethod
    def from_edges(cls, n: int, edges: List[Tuple[int, int]], name: str="", labels: np.ndarray | None=None):
        """Build a graph from a list of vertex pairs.
 
        The input is cleaned before conversion:
 
        * self-loops ``(v, v)`` are dropped;
        * duplicate edges are merged into one, in either orientation (for
          example ``(1, 2)`` and ``(2, 1)``).
 
        Parameters
        ----------
        n : int
            Number of vertices. Edge endpoints must be in ``0 .. n-1``.
        edges : list of (int, int)
            Undirected edges as pairs of 0-based vertex indices.
        name : str, optional
            Instance name to store on the graph.
        labels : np.ndarray, optional
            Original vertex labels to attach. Stored unchanged.
 
        Returns
        -------
        CSRGraph
            The validated graph.
 
        Examples
        --------
        >>> g = CSRGraph.from_edges(3, [(0, 1), (1, 2), (2, 1)], name="path")
        >>> g.m
        2
        >>> neighbors(g, 1)
        array([0, 2], dtype=int32)
        """
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        e = e[e[:, 0] != e[:, 1]]

        lo = np.minimum(e[:, 0], e[:, 1])
        hi = np.maximum(e[:, 0], e[:, 1])
        key = lo * n + hi                      # canonical id per undirected edge
        keep = np.unique(key, return_index=True)[1]
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
        """Build a graph from a NetworkX graph.
 
        Nodes are numbered ``0 .. n-1`` in the order of ``G.nodes``. The
        original node objects are stored in ``labels``.
 
        Parameters
        ----------
        G : networkx.Graph
            Undirected simple graph. Nodes can be any hashable objects.
        name : str, optional
            Instance name to store on the graph.
 
        Returns
        -------
        CSRGraph
            The validated graph, with ``labels`` set.
        """
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
        """Load a graph from an edge-list file.
 
        Parameters
        ----------
        path : str
            Path to the edge-list file.
        name : str, optional
            Instance name to store on the graph.
 
        Returns
        -------
        CSRGraph
            The validated graph. ``labels`` is ``array([0, 1, ..., n-1])``.
        """
        n, edges = read_edge_list(path)
        G = create_networkx_graph(n, edges)
        return cls.from_networkx(G, name=name)


def neighbors(csr_graph: CSRGraph, v: int) -> np.ndarray:
    """Return the neighbours of vertex ``v``.
 
    Parameters
    ----------
    csr_graph : CSRGraph
        The graph.
    v : int
        Vertex index in ``0 .. n-1``.
 
    Returns
    -------
    np.ndarray
        Sorted neighbour indices.
    """
    return csr_graph.nbrs[csr_graph.offsets[v]:csr_graph.offsets[v + 1]]

def degree(csr_graph: CSRGraph, v: int) -> int:
    """Return the degree of vertex ``v``.
 
    Parameters
    ----------
    csr_graph : CSRGraph
        The graph.
    v : int
        Vertex index in ``0 .. n-1``.
 
    Returns
    -------
    int
        Number of neighbours of ``v``, as a NumPy integer.
    """
    return csr_graph.offsets[v + 1] - csr_graph.offsets[v]

def degrees(csr_graph: CSRGraph) -> np.ndarray:
    """Return the degrees of all vertices.
 
    Parameters
    ----------
    csr_graph : CSRGraph
        The graph.
 
    Returns
    -------
    np.ndarray, shape (n,)
        ``degrees[v]`` is the degree of vertex ``v``.
    """
    return np.diff(csr_graph.offsets)

def max_degree(csr_graph: CSRGraph) -> int:
    """Return the maximum vertex degree of the graph.
 
    Parameters
    ----------
    csr_graph : CSRGraph
        The graph. It must have at least one vertex.
 
    Returns
    -------
    int
        The largest degree, as a NumPy integer.
    """
    return degrees(csr_graph).max()


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

def create_networkx_graph(num_nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(range(0, num_nodes))
    G.add_edges_from(edges)
    return G