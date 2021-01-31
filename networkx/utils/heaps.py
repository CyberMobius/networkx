"""
Min-heaps.
"""

from heapq import heappop, heappush
from itertools import count
import networkx as nx
import math

__all__ = ["MinHeap", "PairingHeap", "BinaryHeap", "FibonacciHeap"]


class MinHeap:
    """Base class for min-heaps.

    A MinHeap stores a collection of key-value pairs ordered by their values.
    It supports querying the minimum pair, inserting a new pair, decreasing the
    value in an existing pair and deleting the minimum pair.
    """

    class _Item:
        """Used by subclassess to represent a key-value pair.
        """

        __slots__ = ("key", "value")

        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __repr__(self):
            return repr((self.key, self.value))

    def __init__(self):
        """Initialize a new min-heap.
        """
        self._dict = {}

    def min(self):
        """Query the minimum key-value pair.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        raise NotImplementedError

    def pop(self):
        """Delete the minimum pair in the heap.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        raise NotImplementedError

    def get(self, key, default=None):
        """Returns the value associated with a key.

        Parameters
        ----------
        key : hashable object
            The key to be looked up.

        default : object
            Default value to return if the key is not present in the heap.
            Default value: None.

        Returns
        -------
        value : object.
            The value associated with the key.
        """
        raise NotImplementedError

    def insert(self, key, value, allow_increase=False):
        """Insert a new key-value pair or modify the value in an existing
        pair.

        Parameters
        ----------
        key : hashable object
            The key.

        value : object comparable with existing values.
            The value.

        allow_increase : bool
            Whether the value is allowed to increase. If False, attempts to
            increase an existing value have no effect. Default value: False.

        Returns
        -------
        decreased : bool
            True if a pair is inserted or the existing value is decreased.
        """
        raise NotImplementedError

    def __nonzero__(self):
        """Returns whether the heap if empty.
        """
        return bool(self._dict)

    def __bool__(self):
        """Returns whether the heap if empty.
        """
        return bool(self._dict)

    def __len__(self):
        """Returns the number of key-value pairs in the heap.
        """
        return len(self._dict)

    def __contains__(self, key):
        """Returns whether a key exists in the heap.

        Parameters
        ----------
        key : any hashable object.
            The key to be looked up.
        """
        return key in self._dict


def _inherit_doc(cls):
    """Decorator for inheriting docstrings from base classes.
    """

    def func(fn):
        fn.__doc__ = cls.__dict__[fn.__name__].__doc__
        return fn

    return func


class PairingHeap(MinHeap):
    """A pairing heap.
    """

    class _Node(MinHeap._Item):
        """A node in a pairing heap.

        A tree in a pairing heap is stored using the left-child, right-sibling
        representation.
        """

        __slots__ = ("left", "next", "prev", "parent")

        def __init__(self, key, value):
            super(PairingHeap._Node, self).__init__(key, value)
            # The leftmost child.
            self.left = None
            # The next sibling.
            self.next = None
            # The previous sibling.
            self.prev = None
            # The parent.
            self.parent = None

    def __init__(self):
        """Initialize a pairing heap.
        """
        super().__init__()
        self._root = None

    @_inherit_doc(MinHeap)
    def min(self):
        if self._root is None:
            raise nx.NetworkXError("heap is empty.")
        return (self._root.key, self._root.value)

    @_inherit_doc(MinHeap)
    def pop(self):
        if self._root is None:
            raise nx.NetworkXError("heap is empty.")
        min_node = self._root
        self._root = self._merge_children(self._root)
        del self._dict[min_node.key]
        return (min_node.key, min_node.value)

    @_inherit_doc(MinHeap)
    def get(self, key, default=None):
        node = self._dict.get(key)
        return node.value if node is not None else default

    @_inherit_doc(MinHeap)
    def insert(self, key, value, allow_increase=False):
        node = self._dict.get(key)
        root = self._root
        if node is not None:
            if value < node.value:
                node.value = value
                if node is not root and value < node.parent.value:
                    self._cut(node)
                    self._root = self._link(root, node)
                return True
            elif allow_increase and value > node.value:
                node.value = value
                child = self._merge_children(node)
                # Nonstandard step: Link the merged subtree with the root. See
                # below for the standard step.
                if child is not None:
                    self._root = self._link(self._root, child)
                # Standard step: Perform a decrease followed by a pop as if the
                # value were the smallest in the heap. Then insert the new
                # value into the heap.
                # if node is not root:
                #     self._cut(node)
                #     if child is not None:
                #         root = self._link(root, child)
                #     self._root = self._link(root, node)
                # else:
                #     self._root = (self._link(node, child)
                #                   if child is not None else node)
            return False
        else:
            # Insert a new key.
            node = self._Node(key, value)
            self._dict[key] = node
            self._root = self._link(root, node) if root is not None else node
            return True

    def _link(self, root, other):
        """Link two nodes, making the one with the smaller value the parent of
        the other.
        """
        if other.value < root.value:
            root, other = other, root
        next = root.left
        other.next = next
        if next is not None:
            next.prev = other
        other.prev = None
        root.left = other
        other.parent = root
        return root

    def _merge_children(self, root):
        """Merge the subtrees of the root using the standard two-pass method.
        The resulting subtree is detached from the root.
        """
        node = root.left
        root.left = None
        if node is not None:
            link = self._link
            # Pass 1: Merge pairs of consecutive subtrees from left to right.
            # At the end of the pass, only the prev pointers of the resulting
            # subtrees have meaningful values. The other pointers will be fixed
            # in pass 2.
            prev = None
            while True:
                next = node.next
                if next is None:
                    node.prev = prev
                    break
                next_next = next.next
                node = link(node, next)
                node.prev = prev
                prev = node
                if next_next is None:
                    break
                node = next_next
            # Pass 2: Successively merge the subtrees produced by pass 1 from
            # right to left with the rightmost one.
            prev = node.prev
            while prev is not None:
                prev_prev = prev.prev
                node = link(prev, node)
                prev = prev_prev
            # Now node can become the new root. Its has no parent nor siblings.
            node.prev = None
            node.next = None
            node.parent = None
        return node

    def _cut(self, node):
        """Cut a node from its parent.
        """
        prev = node.prev
        next = node.next
        if prev is not None:
            prev.next = next
        else:
            node.parent.left = next
        node.prev = None
        if next is not None:
            next.prev = prev
            node.next = None
        node.parent = None


class BinaryHeap(MinHeap):
    """A binary heap.
    """

    def __init__(self):
        """Initialize a binary heap.
        """
        super().__init__()
        self._heap = []
        self._count = count()

    @_inherit_doc(MinHeap)
    def min(self):
        dict = self._dict
        if not dict:
            raise nx.NetworkXError("heap is empty")
        heap = self._heap
        pop = heappop
        # Repeatedly remove stale key-value pairs until a up-to-date one is
        # met.
        while True:
            value, _, key = heap[0]
            if key in dict and value == dict[key]:
                break
            pop(heap)
        return (key, value)

    @_inherit_doc(MinHeap)
    def pop(self):
        dict = self._dict
        if not dict:
            raise nx.NetworkXError("heap is empty")
        heap = self._heap
        pop = heappop
        # Repeatedly remove stale key-value pairs until a up-to-date one is
        # met.
        while True:
            value, _, key = heap[0]
            pop(heap)
            if key in dict and value == dict[key]:
                break
        del dict[key]
        return (key, value)

    @_inherit_doc(MinHeap)
    def get(self, key, default=None):
        return self._dict.get(key, default)

    @_inherit_doc(MinHeap)
    def insert(self, key, value, allow_increase=False):
        dict = self._dict
        if key in dict:
            old_value = dict[key]
            if value < old_value or (allow_increase and value > old_value):
                # Since there is no way to efficiently obtain the location of a
                # key-value pair in the heap, insert a new pair even if ones
                # with the same key may already be present. Deem the old ones
                # as stale and skip them when the minimum pair is queried.
                dict[key] = value
                heappush(self._heap, (value, next(self._count), key))
                return value < old_value
            return False
        else:
            dict[key] = value
            heappush(self._heap, (value, next(self._count), key))
            return True


class FibonacciHeap(MinHeap):
    """A Fibonacci heap.

    [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and 
    Clifford Stein. 2009. Introduction to Algorithms, Third Edition (3rd. ed.). 
    The MIT Press., pp.503-526.
    """

    class _FHeapItem(MinHeap._Item):
        """Each member of the fibonacci heap contains a reference to its left
        sibling and right sibling as well as to its parent and one of its
        children. The siblings of a fibonacci heap form a circularly doubly 
        linked list.
        """

        __slots__ = ("left", "right", "child", "parent", "count_children", "mark")

        def __init__(self, key, value):
            """Create an item in the FHeap and mantain references to siblings, 
            parent, and child

            Parameters
            ----------
            key : ValuesView
                A value that corresponds to the value of this item in the heap
            item : Object
                Data that accompanies the key
            """
            super().__init__(key, value)
            self.left: "self._FHeapItem" = self
            self.right: "self._FHeapItem" = self
            self.child: "self._FHeapItem" = None
            self.parent: "self._FHeapItem" = None
            self.count_children: int = 0
            self.mark: bool = False

        def sibling_iterator(self):
            """Iterator over all siblings of this node

            Yields
            -------
            _FHeapItem
                An item that is a sibling of this node
            """
            yield self

            current_node = self.right
            while current_node is not None and current_node is not self:
                yield current_node
                current_node = current_node.right

        def __bool__(self):
            return True

    def __init__(self):
        """Initialize the fibonacci heap with a reference to the min item
        """
        self.min_item: self._FHeapItem
        self.min_item = None

        self.in_set = set()

        self.count = 0

    @_inherit_doc(MinHeap)
    def min(self):
        if self.min_item is None:
            raise nx.NetworkXError("heap is empty.")

        return (self.min_item.key, self.min_item.value)

    @_inherit_doc(MinHeap)
    def pop(self):

        min_item = self.min_item
        self.in_set.remove(min_item)

        # The fibonacci heap only mantains a reference to its minimum item. This
        # minimum item sits in the root list
        if min_item is None:
            raise nx.NetworkXError("heap is empty")

        # Add all the children as siblings to the min, effectively adding them
        # to the root list
        if min_item.child is not None:
            child = self.min_item.child

            # Tie up the pointers and set the parent accordingly
            (min_item.right.left, min_item.right, child.left.right, child.left) = (
                child.left,
                child,
                min_item.right,
                min_item,
            )

            sibling = min_item.right
            while sibling.parent is not None:
                sibling.parent = None

        # If there is no sibling, this is the last element and mark min as None
        if self.min_item.right is self.min_item:
            self.min_item = None

        # Otherwise, make the right sibling as a the new temporary min. Based on
        # the behavior of adding children to the root list this will be a child
        # of the original min item. Then remove the min from the root_list,
        # consolidate the heap, and find the new min item
        else:

            self.min_item.left.right, self.min_item.right.left = (
                self.min_item.right,
                self.min_item.left,
            )

            self.min_item = self.min_item.left
            # Begin consolidating:
            #
            # Cormen at al.: 'It is also where the delayed work of consolidating
            # trees in the root_list finally occurs.' This function is analogous to
            # heapify for other heaps. Look through the root list for two nodes with
            # same degree and make one the child of the other until all nodes in the
            # root list have unique degrees
            #
            # [1] Cormen et. al. creates an array A where A[i] refers to a node in the
            # root list of degree i. Here it's called `degree_list`.
            # When two nodes of equal degree are found, one is made a child of the
            # other.
            # In my implementation specifically, at the time this function is
            # called, self.min_item might not point to the minimum item, but it is
            # guaranteed to be in the root list. We use it to iterate through the root
            # list.
            #
            # Since node degrees are guaranteed to be no bigger than
            # floor(log2(H.n)), we can create an array of appropriate size ahead
            # of time
            degree_list = [None] * self.count.bit_length()

            # Find all elements of our root list
            siblings = [sibling for sibling in self.min_item.sibling_iterator()]

            root_node: self._FHeapItem
            for root_node in siblings:
                x = root_node
                d = x.count_children
                while degree_list[d] is not None:
                    y = degree_list[d]

                    # [1] says exchange these two nodes. I'll swap their keys and
                    # values instead of having to worry about re pointing all their
                    # pointers. Behavior should stay the same
                    if x.value > y.value:
                        x.value, y.value = y.value, x.value
                        x.key, y.key = y.key, x.key
                        x.child, y.child = y.child, x.child

                    # Remove y from the root list and make it a child of x
                    y.mark = False
                    y.parent = x

                    # First, remove it from the root list
                    y.left.right = y.right
                    y.right.left = y.left

                    # Add y as a child of x
                    x.count_children += 1
                    if x.child is None:
                        y.left, y.right = y, y
                        x.child = y
                    else:

                        # Create space for new right node and insert it
                        y.left, y.right = x.child, x.child.right
                        x.child.right.left = y
                        x.child.right = y

                    degree_list[d] = None
                    d += 1

                degree_list[d] = x

            new_min_item = None

            degree_list = [i for i in degree_list if i is not None]

            min_node = degree_list[0]
            for i, node in enumerate(degree_list):
                node.left, degree_list[i - 1].right = degree_list[i - 1], node

                if node.value < min_node.value:
                    min_node = node

            self.min_item = min_node
            # End consolidating

        # Decrease count by one and return
        self.count -= 1
        return (min_item.key, min_item.value)

    @_inherit_doc(MinHeap)
    def insert(self, key, value, allow_increase=False):

        # Since Fibonacci heaps don't mantain any kind of dictionary or mapping
        # to nodes, and it doesn't return the actual FHeap_item, there's no good
        # way to allow_increase
        if allow_increase:
            raise NotImplementedError

        # Create a FHeap item for our key value
        item = self._FHeapItem(key, value)
        min_item = self.min_item

        if item in self.in_set:
            if item.value < self.min_item:
                self.min_item = item

            current_item = item
            while (
                current_item.parent is not current_item
                and current_item.parent.value > current_item.value
            ):
                # Just pipe the key and value up rather than sorting out all the
                # pointers
                (
                    current_item.value,
                    current_item.key,
                    current_item.parent.value,
                    current_item.parent.value,
                ) = (
                    current_item.parent.value,
                    current_item.parent.value,
                    current_item.value,
                    current_item.key,
                )
                current_item = current_item.parent

            return

        # If the heap is empty, make this the new min_item of the heap
        if min_item is None:
            self.min_item = item

        # If the heap has a min item, add the new item to the right of it and
        # compare against the minimum
        else:

            # Create space for new right node and insert it
            item.left, item.right = min_item, min_item.right
            min_item.right.left, min_item.right = item, item

            if value < self.min_item.value:
                self.min_item = item

        self.count += 1
        return item

    def _cut(self, x: _FHeapItem, y: _FHeapItem):
        """Remove x from the child list of y and add it into the root list

        Parameters
        ----------
        x : _FHeapItem
            A member of the child list of y
        y : _FHeapItem
            Some arbitrary _FHeapItem
        """

        # If x is the only item in the child list, set the child list to None
        if x.right is x:
            y.child = None

        # Otherwise, just point the siblings to one another
        else:
            x.right.left, x.left.right = x.left, x.right

        # Add x to the root list
        x.right, x.left = self.min_item, self.min_item.right
        self.min_item.right.left, self.min_item.right = x, x
        x.mark = False
        y.count_children -= 1

    def _cascading_cut(self, y: _FHeapItem):
        """As far as I can tell, cascading cut keeps the tree "pruned" in some
        sense. It prevents nodes from getting too far from the root list and 
        ensures our O(1) run time

        Parameters
        ----------
        y : _FHeapItem
            Some _FHeapItem whose child has been cut
        """

        z = y.parent
        if z is not None:
            if not y.mark:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    def decrease_value(self, x: _FHeapItem, k):
        """Decreases the value of x to k

        Parameters
        ----------
        x : _FHeapItem
            Some arbitrary _FHeap item whose value you want to decrease
        k : typeof(x.value)
            The new value for x.value, must be less than the current value

        Raises
        ------
        nx.NetworkXError
            If the value of k is larger than x's current value
        """
        if k > x.value:
            raise nx.NetworkXError("Key is larger")
        x.value = k
        y = x.parent
        if y is not None and x.value < y.value:
            self._cut(x, y)
            self._cascading_cut(y)
        if x.value < self.min_item.value:
            self.min_item = x

    def union(self, other: "FibonacciHeap"):
        """Adds the elements of other into self 

        Parameters
        ----------
        other : FibonacciHeap
            A Fibonacci heap with items you want to add to self. The other heap
            will be destroyed in the proccess
        """
        self.count += other.count
        (
            self.min_item.right.left,
            self.min_item.right,
            other.min_item.left.right,
            other.min_item.left,
        ) = (other.min_item.left, other.min_item, self.min_item.right, self.min_item)

        if other.min_item.value < self.min_item.value:
            self.min_item = other.min_item

    def delete(self, x: _FHeapItem):
        self.decrease_value(x, -math.inf)
        self.pop()
