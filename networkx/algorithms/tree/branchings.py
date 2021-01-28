"""
Algorithms for finding optimum branchings and spanning arborescences.

This implementation is based on:

    J. Edmonds, Optimum branchings, J. Res. Natl. Bur. Standards 71B (1967),
    233–240. URL: http://archive.org/details/jresv71Bn4p233

"""
# TODO: Implement method from Gabow, Galil, Spence and Tarjan:
#
# @article{
#    year={1986},
#    issn={0209-9683},
#    journal={Combinatorica},
#    volume={6},
#    number={2},
#    doi={10.1007/BF02579168},
#    title={Efficient algorithms for finding minimum spanning trees in
#        undirected and directed graphs},
#    url={https://doi.org/10.1007/BF02579168},
#    publisher={Springer-Verlag},
#    keywords={68 B 15; 68 C 05},
#    author={Gabow, Harold N. and Galil, Zvi and Spencer, Thomas and Tarjan,
#        Robert E.},
#    pages={109-122},
#    language={English}
# }


import string
from itertools import chain, islice
from operator import itemgetter
from typing import Dict, List, Set

import networkx as nx
from networkx.utils import (
    BinaryHeap, CompressedTree, FibonacciHeap, MinHeap, py_random_state)

from .recognition import is_arborescence, is_branching

__all__ = [
    "branching_weight",
    "greedy_branching",
    "maximum_branching",
    "minimum_branching",
    "maximum_spanning_arborescence",
    "minimum_spanning_arborescence",
    "Edmonds",
    "GGST",
]

KINDS = {"max", "min"}

STYLES = {
    "branching": "branching",
    "arborescence": "arborescence",
    "spanning arborescence": "arborescence",
}

INF = float("inf")


@py_random_state(1)
def random_string(L=15, seed=None):
    return "".join([seed.choice(string.ascii_letters) for n in range(L)])


def _min_weight(weight):
    return -weight


def _max_weight(weight):
    return weight


def branching_weight(G, attr="weight", default=1):
    """
    Returns the total weight of a branching.

    """
    return sum(edge[2].get(attr, default) for edge in G.edges(data=True))


@py_random_state(4)
def greedy_branching(G, attr="weight", default=1, kind="max", seed=None):
    """
    Returns a branching obtained through a greedy algorithm.

    This algorithm is wrong, and cannot give a proper optimal branching.
    However, we include it for pedagogical reasons, as it can be helpful to
    see what its outputs are.

    The output is a branching, and possibly, a spanning arborescence. However,
    it is not guaranteed to be optimal in either case.

    Parameters
    ----------
    G : DiGraph
        The directed graph to scan.
    attr : str
        The attribute to use as weights. If None, then each edge will be
        treated equally with a weight of 1.
    default : float
        When `attr` is not None, then if an edge does not have that attribute,
        `default` specifies what value it should take.
    kind : str
        The type of optimum to search for: 'min' or 'max' greedy branching.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    B : directed graph
        The greedily obtained branching.

    """
    if kind not in KINDS:
        raise nx.NetworkXException("Unknown value for `kind`.")

    if kind == "min":
        reverse = False
    else:
        reverse = True

    if attr is None:
        # Generate a random string the graph probably won't have.
        attr = random_string(seed=seed)

    edges = [(u, v, data.get(attr, default)) for (u, v, data) in G.edges(data=True)]

    # We sort by weight, but also by nodes to normalize behavior across runs.
    try:
        edges.sort(key=itemgetter(2, 0, 1), reverse=reverse)
    except TypeError:
        # This will fail in Python 3.x if the nodes are of varying types.
        # In that case, we use the arbitrary order.
        edges.sort(key=itemgetter(2), reverse=reverse)

    # The branching begins with a forest of no edges.
    B = nx.DiGraph()
    B.add_nodes_from(G)

    # Now we add edges greedily so long we maintain the branching.
    uf = nx.utils.UnionFind()
    for i, (u, v, w) in enumerate(edges):
        if uf[u] == uf[v]:
            # Adding this edge would form a directed cycle.
            continue
        elif B.in_degree(v) == 1:
            # The edge would increase the degree to be greater than one.
            continue
        else:
            # If attr was None, then don't insert weights...
            data = {}
            if attr is not None:
                data[attr] = w
            B.add_edge(u, v, **data)
            uf.union(u, v)

    return B


class MultiDiGraph_EdgeKey(nx.MultiDiGraph):
    """
    MultiDiGraph which assigns unique keys to every edge.

    Adds a dictionary edge_index which maps edge keys to (u, v, data) tuples.

    This is not a complete implementation. For Edmonds algorithm, we only use
    add_node and add_edge, so that is all that is implemented here. During
    additions, any specified keys are ignored---this means that you also
    cannot update edge attributes through add_node and add_edge.

    Why do we need this? Edmonds algorithm requires that we track edges, even
    as we change the head and tail of an edge, and even changing the weight
    of edges. We must reliably track edges across graph mutations.

    """

    def __init__(self, incoming_graph_data=None, **attr):
        cls = super()
        cls.__init__(incoming_graph_data=incoming_graph_data, **attr)

        self._cls = cls
        self.edge_index = {}

    def remove_node(self, n):
        keys = set()
        for keydict in self.pred[n].values():
            keys.update(keydict)
        for keydict in self.succ[n].values():
            keys.update(keydict)

        for key in keys:
            del self.edge_index[key]

        self._cls.remove_node(n)

    def remove_nodes_from(self, nbunch):
        for n in nbunch:
            self.remove_node(n)

    def add_edge(self, u_for_edge, v_for_edge, key_for_edge, **attr):
        """
        Key is now required.

        """
        u, v, key = u_for_edge, v_for_edge, key_for_edge
        if key in self.edge_index:
            uu, vv, _ = self.edge_index[key]
            if (u != uu) or (v != vv):
                raise Exception(f"Key {key!r} is already in use.")

        self._cls.add_edge(u, v, key, **attr)
        self.edge_index[key] = (u, v, self.succ[u][v][key])

    def add_edges_from(self, ebunch_to_add, **attr):
        for u, v, k, d in ebunch_to_add:
            self.add_edge(u, v, k, **d)

    def remove_edge_with_key(self, key):
        try:
            u, v, _ = self.edge_index[key]
        except KeyError as e:
            raise KeyError(f"Invalid edge key {key!r}") from e
        else:
            del self.edge_index[key]
            self._cls.remove_edge(u, v, key)

    def remove_edges_from(self, ebunch):
        raise NotImplementedError


def get_path(G, u, v):
    """
    Returns the edge keys of the unique path between u and v.

    This is not a generic function. G must be a branching and an instance of
    MultiDiGraph_EdgeKey.

    """
    nodes = nx.shortest_path(G, u, v)
    # We are guaranteed that there is only one edge connected every node
    # in the shortest path.

    def first_key(i, vv):
        # Needed for 2.x/3.x compatibilitity
        keys = G[nodes[i]][vv].keys()
        # Normalize behavior
        keys = list(keys)
        return keys[0]

    edges = [first_key(i, vv) for i, vv in enumerate(nodes[1:])]
    return nodes, edges


class Edmonds:
    """
    Edmonds algorithm for finding optimal branchings and spanning arborescences.

    """

    def __init__(self, G, seed=None):
        self.G_original = G

        # Need to fix this. We need the whole tree.
        self.store = True

        # The final answer.
        self.edges = []

        # Since we will be creating graphs with new nodes, we need to make
        # sure that our node names do not conflict with the real node names.
        self.template = random_string(seed=seed) + "_{0}"

    def _init(self, attr, default, kind, style, preserve_attrs, seed):
        if kind not in KINDS:
            raise nx.NetworkXException("Unknown value for `kind`.")

        # Store inputs.
        self.attr = attr
        self.default = default
        self.kind = kind
        self.style = style

        # Determine how we are going to transform the weights.
        if kind == "min":
            self.trans = trans = _min_weight
        else:
            self.trans = trans = _max_weight

        if attr is None:
            # Generate a random attr the graph probably won't have.
            attr = random_string(seed=seed)

        # This is the actual attribute used by the algorithm.
        self._attr = attr

        # This attribute is used to store whether a particular edge is still
        # a candidate. We generate a random attr to remove clashes with
        # preserved edges
        self.candidate_attr = "candidate_" + random_string(seed=seed)

        # The object we manipulate at each step is a multidigraph.
        self.G = G = MultiDiGraph_EdgeKey()
        for key, (u, v, data) in enumerate(self.G_original.edges(data=True)):
            d = {attr: trans(data.get(attr, default))}

            if preserve_attrs:
                for (d_k, d_v) in data.items():
                    if d_k != attr:
                        d[d_k] = d_v

            G.add_edge(u, v, key, **d)

        self.level = 0

        # These are the "buckets" from the paper.
        #
        # As in the paper, G^i are modified versions of the original graph.
        # D^i and E^i are nodes and edges of the maximal edges that are
        # consistent with G^i. These are dashed edges in figures A-F of the
        # paper. In this implementation, we store D^i and E^i together as a
        # graph B^i. So we will have strictly more B^i than the paper does.
        self.B = MultiDiGraph_EdgeKey()
        self.B.edge_index = {}
        self.graphs = []  # G^i
        self.branchings = []  # B^i
        self.uf = nx.utils.UnionFind()

        # A list of lists of edge indexes. Each list is a circuit for graph G^i.
        # Note the edge list will not, in general, be a circuit in graph G^0.
        self.circuits = []
        # Stores the index of the minimum edge in the circuit found in G^i and B^i.
        # The ordering of the edges seems to preserve the weight ordering from G^0.
        # So even if the circuit does not form a circuit in G^0, it is still true
        # that the minimum edge of the circuit in G^i is still the minimum edge
        # in circuit G^0 (depsite their weights being different).
        self.minedge_circuit = []

    def find_optimum(
        self,
        attr="weight",
        default=1,
        kind="max",
        style="branching",
        preserve_attrs=False,
        seed=None,
    ):
        """
        Returns a branching from G.

        Parameters
        ----------
        attr : str
            The edge attribute used to in determining optimality.
        default : float
            The value of the edge attribute used if an edge does not have
            the attribute `attr`.
        kind : {'min', 'max'}
            The type of optimum to search for, either 'min' or 'max'.
        style : {'branching', 'arborescence'}
            If 'branching', then an optimal branching is found. If `style` is
            'arborescence', then a branching is found, such that if the
            branching is also an arborescence, then the branching is an
            optimal spanning arborescences. A given graph G need not have
            an optimal spanning arborescence.
        preserve_attrs : bool
            If True, preserve the other edge attributes of the original
            graph (that are not the one passed to `attr`)
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        H : (multi)digraph
            The branching.

        """
        self._init(attr, default, kind, style, preserve_attrs, seed)
        uf = self.uf

        # This enormous while loop could use some refactoring...

        G, B = self.G, self.B
        D = set()
        nodes = iter(list(G.nodes()))
        attr = self._attr
        G_pred = G.pred

        def desired_edge(v):
            """
            Find the edge directed toward v with maximal weight.

            """
            edge = None
            weight = -INF
            for u, _, key, data in G.in_edges(v, data=True, keys=True):
                new_weight = data[attr]
                if new_weight > weight:
                    weight = new_weight
                    edge = (u, v, key, new_weight)

            return edge, weight

        while True:
            # (I1): Choose a node v in G^i not in D^i.
            try:
                v = next(nodes)
            except StopIteration:
                # If there are no more new nodes to consider, then we *should*
                # meet the break condition (b) from the paper:
                #   (b) every node of G^i is in D^i and E^i is a branching
                # Construction guarantees that it's a branching.
                assert len(G) == len(B)
                if len(B):
                    assert is_branching(B)

                if self.store:
                    self.graphs.append(G.copy())
                    self.branchings.append(B.copy())

                    # Add these to keep the lengths equal. Element i is the
                    # circuit at level i that was merged to form branching i+1.
                    # There is no circuit for the last level.
                    self.circuits.append([])
                    self.minedge_circuit.append(None)
                break
            else:
                if v in D:
                    # print("v in D", v)
                    continue

            # Put v into bucket D^i.
            # print(f"Adding node {v}")
            D.add(v)
            B.add_node(v)

            edge, weight = desired_edge(v)
            # print(f"Max edge is {edge!r}")
            if edge is None:
                # If there is no edge, continue with a new node at (I1).
                continue
            else:
                # Determine if adding the edge to E^i would mean its no longer
                # a branching. Presently, v has indegree 0 in B---it is a root.
                u = edge[0]

                if uf[u] == uf[v]:
                    # Then adding the edge will create a circuit. Then B
                    # contains a unique path P from v to u. So condition (a)
                    # from the paper does hold. We need to store the circuit
                    # for future reference.
                    Q_nodes, Q_edges = get_path(B, v, u)
                    Q_edges.append(edge[2])
                else:
                    # Then B with the edge is still a branching and condition
                    # (a) from the paper does not hold.
                    Q_nodes, Q_edges = None, None

                # Conditions for adding the edge.
                # If weight < 0, then it cannot help in finding a maximum branching.
                if self.style == "branching" and weight <= 0:
                    acceptable = False
                else:
                    acceptable = True

                # print(f"Edge is acceptable: {acceptable}")
                if acceptable:
                    dd = {attr: weight}
                    B.add_edge(u, v, edge[2], **dd)
                    G[u][v][edge[2]][self.candidate_attr] = True
                    uf.union(u, v)
                    if Q_edges is not None:
                        # print("Edge introduced a simple cycle:")
                        # print(Q_nodes, Q_edges)

                        # Move to method
                        # Previous meaning of u and v is no longer important.

                        # Apply (I2).
                        # Get the edge in the cycle with the minimum weight.
                        # Also, save the incoming weights for each node.
                        minweight = INF
                        minedge = None
                        Q_incoming_weight = {}
                        for edge_key in Q_edges:
                            u, v, data = B.edge_index[edge_key]
                            w = data[attr]
                            Q_incoming_weight[v] = w
                            if w < minweight:
                                minweight = w
                                minedge = edge_key

                        self.circuits.append(Q_edges)
                        self.minedge_circuit.append(minedge)

                        if self.store:
                            self.graphs.append(G.copy())
                        # Always need the branching with circuits.
                        self.branchings.append(B.copy())

                        # Now we mutate it.
                        new_node = self.template.format(self.level)

                        # print(minweight, minedge, Q_incoming_weight)

                        G.add_node(new_node)
                        new_edges = []
                        for u, v, key, data in G.edges(data=True, keys=True):
                            if u in Q_incoming_weight:
                                if v in Q_incoming_weight:
                                    # Circuit edge, do nothing for now.
                                    # Eventually delete it.
                                    continue
                                else:
                                    # Outgoing edge. Make it from new node
                                    dd = data.copy()
                                    new_edges.append((new_node, v, key, dd))
                            else:
                                if v in Q_incoming_weight:
                                    # Incoming edge. Change its weight
                                    w = data[attr]
                                    w += minweight - Q_incoming_weight[v]
                                    dd = data.copy()
                                    dd[attr] = w
                                    new_edges.append((u, new_node, key, dd))
                                else:
                                    # Outside edge. No modification necessary.
                                    continue

                        G.remove_nodes_from(Q_nodes)
                        B.remove_nodes_from(Q_nodes)
                        D.difference_update(set(Q_nodes))

                        for u, v, key, data in new_edges:
                            G.add_edge(u, v, key, **data)
                            if self.candidate_attr in data:
                                del data[self.candidate_attr]
                                B.add_edge(u, v, key, **data)
                                uf.union(u, v)

                        nodes = iter(list(G.nodes()))
                        self.level += 1

        # (I3) Branch construction.
        # print(self.level)
        H = self.G_original.__class__()

        def is_root(G, u, edgekeys):
            """
            Returns True if `u` is a root node in G.

            Node `u` will be a root node if its in-degree, restricted to the
            specified edges, is equal to 0.

            """
            if u not in G:
                # print(G.nodes(), u)
                raise Exception(f"{u!r} not in G")
            for v in G.pred[u]:
                for edgekey in G.pred[u][v]:
                    if edgekey in edgekeys:
                        return False, edgekey
            else:
                return True, None

        # Start with the branching edges in the last level.
        edges = set(self.branchings[self.level].edge_index)
        while self.level > 0:
            self.level -= 1

            # The current level is i, and we start counting from 0.

            # We need the node at level i+1 that results from merging a circuit
            # at level i. randomname_0 is the first merged node and this
            # happens at level 1. That is, randomname_0 is a node at level 1
            # that results from merging a circuit at level 0.
            merged_node = self.template.format(self.level)

            # The circuit at level i that was merged as a node the graph
            # at level i+1.
            circuit = self.circuits[self.level]
            # print
            # print(merged_node, self.level, circuit)
            # print("before", edges)
            # Note, we ask if it is a root in the full graph, not the branching.
            # The branching alone doesn't have all the edges.

            isroot, edgekey = is_root(self.graphs[self.level + 1], merged_node, edges)
            edges.update(circuit)
            if isroot:
                minedge = self.minedge_circuit[self.level]
                if minedge is None:
                    raise Exception

                # Remove the edge in the cycle with minimum weight.
                edges.remove(minedge)
            else:
                # We have identified an edge at next higher level that
                # transitions into the merged node at the level. That edge
                # transitions to some corresponding node at the current level.
                # We want to remove an edge from the cycle that transitions
                # into the corresponding node.
                # print("edgekey is: ", edgekey)
                # print("circuit is: ", circuit)
                # The branching at level i
                G = self.graphs[self.level]
                # print(G.edge_index)
                target = G.edge_index[edgekey][1]
                for edgekey in circuit:
                    u, v, data = G.edge_index[edgekey]
                    if v == target:
                        break
                else:
                    raise Exception("Couldn't find edge incoming to merged node.")
                # print(f"not a root. removing {edgekey}")

                edges.remove(edgekey)

        self.edges = edges

        H.add_nodes_from(self.G_original)
        for edgekey in edges:
            u, v, d = self.graphs[0].edge_index[edgekey]
            dd = {self.attr: self.trans(d[self.attr])}

            # Optionally, preserve the other edge attributes of the original
            # graph
            if preserve_attrs:
                for (key, value) in d.items():
                    if key not in [self.attr, self.candidate_attr]:
                        dd[key] = value

            # TODO: make this preserve the key.
            H.add_edge(u, v, **dd)

        return H


class GGST:
    """
    Gabow, Galil, Spence and Tarjan's algorithm for finding optimal branchings 
    and spanning arborescences. 

    A brief description of the algorithm: Start by creating an empty 'growth path'. To
    start, add an arbitrary node into it. This node is now the 'head' of our growth 
    path. Continue by adding the minimum weight parent of the head node to the growth
    path, making it the new head. Continue adding minimum weight parents of the head
    node to the growth path until either all nodes are exhausted or you create a cycle
    with your growth path. 
    If all nodes are exhausted, this is the spanning arborescence, return it. Otherwise
    compress all the nodes that form the cycle into one node, and contine with our
    process of adding to the growth path until we've exhausted all the nodes.
    Then for each compressed node decompress them and choose some edge from the cycle to 
    remove. Once all cycles are removed we should be left with the optimal spanning 
    arborescence.
    
    [1] Gabow, H.N., Galil, Z., Spencer, T. et al. Efficient algorithms for 
    finding minimum spanning trees in undirected and directed graphs. 
    Combinatorica 6, 109–122 (1986). https://doi.org/10.1007/BF02579168
    """

    class _CandidateEdges:
        """From the paper[1]: "We need some lists and sets to keep track of candidate 
        edges and to allow easy deletion of multiple edges. For each vertex v not on the 
        growth not on the growth path we maintain a an exit list of the current edges 
        (v, v_i), sorted in increasing order on i. (Recall that the growth path contains
        vertices v_0,v_1,...,v_l.) For each vertex v_j on the growth path we maintain a 
        similar exit list, of those edges (v_j, v_i) with j < i also sorted in 
        increasing order on i. Among those edges in the exit list, the first (of lowest
        index i) is active; the others are passive. For each vertex v_i, we maintain a 
        passive set containing all passive edges entering v_i"
        """
        exit_list: List
        passive_set: Set

        def __init__(self, exit_list = None, passive_set = None):
            self.exit_list = exit_list if exit_list is not None else []
            self.passive_set = passive_set if passive_set is not None else {}


    def __init__(
        self, G: nx.DiGraph, 
        attr="weight",
        default=1,
        kind="max"
        ):
        self.G = G
        self.default = default
        self.attr = attr
        self.kind = kind

        self._edge_mapping = {}

        # Fibonacci heaps are really slow, on my computer I haven't found any size on
        # which fibonacci heaps are faster in practice. I'll set some threshold at about
        # 1 billion where we'll switch over heaps. 
        if len(G) < 1000000000:
            self.heap = BinaryHeap

        else:
            self.heap = FibonacciHeap

    def _get_edge(self, node_from, node_to) -> "Edge":
        if (node_from, node_to) in self._edge_mapping:
            return self._edge_mapping[(node_from, node_to)]
        return self.G[node_from][node_to]

    def _add_edge(node_from, node_to):
        pass

    def find_optimum(
        self,
        style="branching",
        preserve_attrs=False,
        seed=None,
    ):
        """
        Returns a branching from G.

        Parameters
        ----------
        attr : str
            The edge attribute used to in determining optimality.
        default : float
            The value of the edge attribute used if an edge does not have
            the attribute `attr`.
        kind : {'min', 'max'}
            The type of optimum to search for, either 'min' or 'max'.
        style : {'branching', 'arborescence'}
            If 'branching', then an optimal branching is found. If `style` is
            'arborescence', then a branching is found, such that if the
            branching is also an arborescence, then the branching is an
            optimal spanning arborescences. A given graph G need not have
            an optimal spanning arborescence.
        preserve_attrs : bool
            If True, preserve the other edge attributes of the original
            graph (that are not the one passed to `attr`)
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        H : (multi)digraph
            The branching.

        """
        # The Gabow, Galil, Spence, Tarjan algorithm is mostly an optimization
        # of Edmond's algorithm above.
        #
        # Firstly, we assume that our graph is complete, if it isn't we'll simply
        # add edges of infinite weight between each vertex. As an implementation
        # detail, we won't actually create all these edges upfront, but if we
        # ever get "stuck," we'll create an infinite weight edge to some
        # arbitrary vertex.
        #
        # Secondly, we pick an arbitrary vertex to start from, call it s.
        # Pick the minimum weight edge from the rest of the graph to s. This
        # edge, (v,s), will be added to the set of edges we're considering for
        # our minimum arborescence. Now, add the minimum weight edge from the
        # rest of the graph connecting the parent of s, this v vertex.
        # Similarly, add this edge to the set of edges we're considering for our
        # spanning arborescence. We keep adding minimum weight edges to the
        # root of our growing arborescence. If this growing arborescence spans
        # all vertices, we can return it.
        # However, if we create a loop, we'll replace all the edges of the self
        # loop with
        # TODO: Finish this explanation

        G = self.G
        attr = self.attr
        selected_nodes = CompressedTree()

        # Initialize our growth path
        growth_path = {G.nodes[0]: 0}
        growth_path_list = [G.nodes[0]]
        growth_path_index = 1

        node_heaps: Dict[object, MinHeap] = {}

        head = selected_nodes[G.nodes[0]]

        # Initialize our exit lists and passive sets
        canditdate_edges: Dict[object, self._CandidateEdges]
        canditdate_edges = {node: self._CandidateEdges() for node in G.nodes}
        # For all edges of the form (v, head) add the edge to the exist list of 
        for v in G.predecessors(head):
            canditdate_edges[v].exit_list.append(head)

        while True:
            try:
                new_head = min(
                    p
                    for p in G.predecessors(head) 
                    if p is not head, 
                    key=lambda x: G[x][head][attr]
                )

            except (StopIteration, ValueError):
                # Like I described above, if we can't find any predecessors,
                # we'll create an edge with infinite weight to some node.
                # To hopefully improve on performance, we'll pick one that
                # doesn't immediately create a loop. For the time being this is
                # an implicit edge.
                new_head = next(x for x in G.nodes if x not in growth_path)

            new_head = selected_nodes[new_head]

            if new_head in growth_path:
                # Compress loop into one node. This is case 2 from the paper [1]

                # TODO see if there's a way to avoid creating this intermediate list
                # to free up some space for potentially large loops
                loop_start = growth_path[new_head]
                loop_nodes = growth_path_list[loop_start:]

                # The change in value needs to happen simultaneously for all nodes, so
                # store the change in value we want to induce before applying
                value_changes = []
                for i in range(-1, len(loop_nodes)-1):
                    this_node, next_node = loop_nodes[i], loop_nodes[i+1]
                    edge_cost = self._get_edge(this_node, next_node)[attr]

                    delta = -1 * edge_cost - selected_nodes.get_value(next_node)
                    
                    value_changes.append(delta)
                
                # Apply the delta for every node we're compressing
                for i in range(len(value_changes)):
                    selected_nodes.change_value(value_changes[i], loop_nodes[i])

                for v_i in loop_nodes:
                    passive_edges  = canditdate_edges[v_i].passive_set
                    for j in range(len(passive_edges)-1):
                        x = passive_edges[j]
                        max_edge = max(
                            [
                                v_i,
                                passive_edges[-1]

                            ],
                            key = lambda n: G[x][n]['weight']
                        )

                        canditdate_edges[x].exit_list.remove(max_edge)


                representative = selected_nodes.union(*loop_nodes)
                canditdate_edges[representative] = self._CandidateEdges()

            else:
                # This corresponds to case 1 from the paper[1]
                growth_path_list.append(new_head)
                growth_path_index += 1
                growth_path[new_head] = growth_path_index
                selected_nodes.union(new_head)

                node_heaps[new_head] = self.heap()

                for v in canditdate_edges[new_head].exit_list:
                    # new_head corresponds to the u vertex from the paper
                    canditdate_edges[v].passive_set.remove((new_head,v))
                
                for x in G.predecessors(new_head):

                    if not canditdate_edges[x].exit_list:
                        node_heaps[new_head].insert(x, G[x][new_head][attr])


                    try:
                        canditdate_edges[new_head].passive_set.add(
                            canditdate_edges[x].exit_list[-1]
                        )
                    except IndexError:
                        pass

                    canditdate_edges[x].exit_list.append(new_head)
                    

            head = new_head


def maximum_branching(G, attr="weight", default=1, preserve_attrs=False):
    ed = Edmonds(G)
    B = ed.find_optimum(
        attr, default, kind="max", style="branching", preserve_attrs=preserve_attrs
    )
    return B


def minimum_branching(G, attr="weight", default=1, preserve_attrs=False):
    ed = Edmonds(G)
    B = ed.find_optimum(
        attr, default, kind="min", style="branching", preserve_attrs=preserve_attrs
    )
    return B


def maximum_spanning_arborescence(G, attr="weight", default=1, preserve_attrs=False):
    ed = Edmonds(G)
    B = ed.find_optimum(
        attr, default, kind="max", style="arborescence", preserve_attrs=preserve_attrs
    )
    if not is_arborescence(B):
        msg = "No maximum spanning arborescence in G."
        raise nx.exception.NetworkXException(msg)
    return B


def minimum_spanning_arborescence(G, attr="weight", default=1, preserve_attrs=False):
    ed = Edmonds(G)
    B = ed.find_optimum(
        attr, default, kind="min", style="arborescence", preserve_attrs=preserve_attrs
    )
    if not is_arborescence(B):
        msg = "No minimum spanning arborescence in G."
        raise nx.exception.NetworkXException(msg)
    return B


docstring_branching = """
Returns a {kind} {style} from G.

Parameters
----------
G : (multi)digraph-like
    The graph to be searched.
attr : str
    The edge attribute used to in determining optimality.
default : float
    The value of the edge attribute used if an edge does not have
    the attribute `attr`.
preserve_attrs : bool
    If True, preserve the other attributes of the original graph (that are not
    passed to `attr`)

Returns
-------
B : (multi)digraph-like
    A {kind} {style}.
"""

docstring_arborescence = (
    docstring_branching
    + """
Raises
------
NetworkXException
    If the graph does not contain a {kind} {style}.

"""
)

maximum_branching.__doc__ = docstring_branching.format(
    kind="maximum", style="branching"
)

minimum_branching.__doc__ = docstring_branching.format(
    kind="minimum", style="branching"
)

maximum_spanning_arborescence.__doc__ = docstring_arborescence.format(
    kind="maximum", style="spanning arborescence"
)

minimum_spanning_arborescence.__doc__ = docstring_arborescence.format(
    kind="minimum", style="spanning arborescence"
)
