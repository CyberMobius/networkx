"""
Union-find data structure.
"""

from networkx.utils import groups


class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """

    def __init__(self, elements=None):
        """Create a new empty union-find structure.

        If *elements* is an iterable, this structure will be initialized
        with the discrete partition on the given set of elements.

        """
        if elements is None:
            elements = ()
        self.parents = {}
        self.weights = {}
        for x in elements:
            self.weights[x] = 1
            self.parents[x] = x

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def to_sets(self):
        """Iterates over the sets stored in this structure.

        For example::

            >>> partition = UnionFind("xyz")
            >>> sorted(map(sorted, partition.to_sets()))
            [['x'], ['y'], ['z']]
            >>> partition.union("x", "y")
            >>> sorted(map(sorted, partition.to_sets()))
            [['x', 'y'], ['z']]

        """
        # Ensure fully pruned paths
        for x in self.parents.keys():
            _ = self[x]  # Evaluated for side-effect only

        yield from groups(self.parents).values()

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        # Find the heaviest root according to its weight.
        roots = iter(sorted({self[x] for x in objects}, key=lambda r: self.weights[r]))
        try:
            root = next(roots)
        except StopIteration:
            return

        for r in roots:
            self.weights[root] += self.weights[r]
            self.parents[r] = root


class CompressedTree(UnionFind):
    """Compressed Tree datatype for storing disjoint sets. It's similar to the
    UnionFind datatype above with one exception. Each value has an associated
    real number and the value of some node is stored as it's difference from the
    parent.
    So if we have two nodes of value 5 and 8, then we make the 5 node the child
    of the 8 node, we'll store the 5 as -3, (5 - 8 = -3). This way, we can
    recover the value by adding the value we have stored to all of it's parents.
    We store it this way so that we can implement change_value(self, delta, A).
    This way when we add delta to the value of the set A, all it's children get
    delta added to them as well for free. This is a useful property for some
    graph algorithms.

    [1]  Michael L. Fredman and Robert Endre Tarjan. 1987. Fibonacci heaps and
    their uses in improved network optimization algorithms. J. ACM 34, 3
    (July 1987), 596–615. DOI:https://doi.org/10.1145/28869.28874

    [2] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford
    Stein. 2009. Introduction to Algorithms, Third Edition (3rd. ed.). The MIT
    Press.

    [3] The code above for UnionFind
    """

    def __init__(self, elements=None) -> None:
        """Create a new empty union-find structure

        Parameters
        ----------
        elements : iter, optional
            If *elements* is an iterable, this structure will be initialized
            with the discrete partition on the given set of elements.,
            by default None
        """
        super().__init__(elements)
        self.value_dict = {element: 0 for element in elements}

    def change_value(self, delta, A):
        """Add delta to the value for A

        Parameters
        ----------
        delta : number
            The number we want to add to the attribute of A we specified in the
            __init__
        A : Any
            An object from the set
        """
        self.value_dict[A] += delta

    def union(self, *objects):
        """Find the sets containing the objects and merge them all.

        Parameters
        ----------
        objects : iterable
            A list of objects that have the self.attribute attribute

        Raises
        ------
        AttributeError
            If any of the objects don't have the attribute we provided in the
            __init__, then raise this error.
        """
        roots = iter(sorted({self[x] for x in objects}, key=lambda r: self.weights[r]))
        try:
            root = next(roots)
        except StopIteration:
            return

        for r in roots:
            self.weights[root] += self.weights[r]
            self.parents[r] = root
            self.value_dict[r] -= self.value_dict[root]

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            self.value_dict[object] = 0
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]

        value_stack = [0]

        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
            value_stack.append(self.value_dict[root] + value_stack[-1])

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
            self.value_dict[ancestor] = self.value_dict[ancestor] + value_stack.pop()
        return root

    def get_value(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            self.value_dict[object] = 0
            return 0

        # Find the name of the set this object belongs to
        parent_object = self[object]

        if parent_object is object:
            return self.value_dict[object]

        return self.value_dict[parent_object] + self.value_dict[object]

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        # Find the heaviest root according to its weight.
        roots = iter(sorted({self[x] for x in objects}, key=lambda r: self.weights[r]))
        try:
            root = next(roots)
        except StopIteration:
            return

        for r in roots:
            self.weights[root] += self.weights[r]
            self.parents[r] = root
            self.value_dict[r] -= self.value_dict[root]

        return root