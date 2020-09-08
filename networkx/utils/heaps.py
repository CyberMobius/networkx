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
            """Create 

            Parameters
            ----------
            key : [type]
                [description]
            item : [type]
                [description]
            """
            super().__init__(key, value)
            self.left: _FHeapItem = None
            self.right: _FHeapItem = None
            self.child: _FHeapItem = None
            self.parent: _FHeapItem = None
            self.count_children: int = 0
            self.mark: bool = False

        def _set_siblings(self, left, right):
            """A shorthand way to set the siblings of an item. This does NOT 
            preserve the doubly linked list

            Parameters
            ----------
            left : _FHeapItem
                The item left of this one in the FHeap
            right : _FHeapItem
                The item left of this one in the FHeap
            """
            self.left = left
            self.right = right

        def add_right(self, right_node):
            """Adds an item to the right of this node

            Parameters
            ----------
            right_node : _FHeapItem
                The node to go right of this one
            """
            # If there is no right node i.e. the linked list has only one
            # element
            if self.right is None:
                self._set_siblings(right_node, right_node)
                right_node._set_siblings(self, self)

            # Create space for new right node
            else:
                right_node._set_siblings(self, self.right)
                self.right.left = right_node
                self.right = right_node

        def add_left(self, left_node):
            """Adds an item to the left of this node

            Parameters
            ----------
            left_node : _FHeapItem
                The node to go left of this one
            """
            # If there are no siblings, then add it to the left, and since this
            # is a circular doubly linked list
            if self.left is None:
                self._set_siblings(left_node, left_node)
                left_node._set_siblings(self, self)
            else:
                self.left.add_right(left_node)

        def add_child(self, new_child):
            """Add new Node as child to this node

            Parameters
            ----------
            new_child : _FHeapItem
                Node that will become the child of this node
            """
            self.count_children += 1
            if self.child is None:
                new_child._set_siblings(None, None)
                self.child = new_child
            else:
                self.child.add_right(new_child)

        def sibling_iterator(self):
            """Iterator over all siblings of the self node

            Yields
            -------
            _FHeapItem
                An item that is a sibling of this node
            """
            yield self

            current_node = self.right
            while current_node is not None and current_node != self:
                yield current_node
                current_node = current_node.right

    def __init__(self):
        """Initialize the fibonacci heap with a reference to the min item
        """
        self.min_item: self._FHeapItem
        self.min_item = None

        self.count = 0
        self.phi = phi = (1 + math.sqrt(5)) / 2

    @_inherit_doc(MinHeap)
    def min(self):
        if self.min_item is None:
            raise nx.NetworkXError("heap is empty.")

        return (self.min_item.key, self.min_item.value)

    def _add_children_to_root_list(self):
        """Helper function to add children to the root list
        """
        if self.min_item.child is None:
            return

        # Keep track of a child
        child = self.min_item.child

        # If this child is the only element of the list simply add it as a
        # sibling to the min item. Mark it as not having a parent
        if child.right is None:
            child.parent = None
            self.min_item.child = None
            self.min_item.add_right(child)

        # For element in the child linked list, add it as a sibling to the min
        # item
        else:
            # Start from the first child
            siblings = [sibling for sibling in child.sibling_iterator()]
            for sibling in siblings:
                sibling.parent = None
                self.min_item.add_right(sibling)

    @_inherit_doc(MinHeap)
    def pop(self):

        # The fibonacci heap only mantains a reference to its minimum item. This
        # minimum item sits in the
        if self.min_item is None:
            raise nx.NetworkXError("heap is empty")

        # Add all the children as siblings to the min, effectively adding them
        # to the root_list
        self._add_children_to_root_list()

        # Keep track of the min item
        min_item = self.min_item

        # If there is no sibling, this is the last element and mark min as None
        if self.min_item.right is None:
            self.min_item = None

        # Otherwise, make the right sibling as a the new temporary min. Based on
        # the behavior of self._add_children_to_root_list this will be a child
        # of the original min item. Then consolidate the heap and find the new
        # min
        else:
            if self.min_item.left != self.min_item.right:
                self.min_item.left.right, self.min_item.right.left = (
                    self.min_item.right,
                    self.min_item.left,
                )

            else:
                self.min_item.left._set_siblings(None, None)

            self.min_item = self.min_item.left
            self._consolidate()

        # Decrease count by one and return
        self.count -= 1
        return (min_item.key, min_item.value)

    def _consolidate(self):
        """Cormen at al.: 'It is also where the delayed work of consolidating
        trees in the root_list finally occurs.' This function is analogous to 
        heapify for other heaps. Look through the root list for two nodes with 
        same degree and make one the child of the other until all nodes in the 
        root list have unique degrees

        [1] Cormen et. al. creates an array A where A[i] refers to a node in the
        root list of degree i. Here it's called `degree_list`. 
        When two nodes of equal degree are found, one is made a child of the 
        other. 
        In my implementation specifically, at the time this function is
        called, self.min_item might not point to the minimum item, but it is
        guaranteed to be in the root list. We use it to iterate through the root
        list.
        """

        # Since node degrees are guaranteed to be no bigger than
        # floor(log_phi(H.n)), we can create an array of appropriate size ahead
        # of time
        degree_list = [None for i in range(1 + int(math.log(self.count, self.phi)))]

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

                self._FibHeapLink(y, x)
                degree_list[d] = None
                d += 1

            degree_list[d] = x

        self.min_item = None
        for i in range(len(degree_list)):
            if degree_list[i] is not None:
                if self.min_item is None:
                    self.min_item = degree_list[i]
                    self.min_item._set_siblings(None, None)
                else:
                    self.min_item.add_right(degree_list[i])
                    if self.min_item.value > degree_list[i].value:
                        self.min_item = degree_list[i]

    def _FibHeapLink(self, y: _FHeapItem, x: _FHeapItem):
        """Remove y from the root list and make it a child of x

        Parameters
        ----------
        y : self._FHeapItem
            The larger of the two elements
        x : self._FHeapItem
            The smaller of the two elements, it stays in the root list and y
            becomes its child
        """
        y.mark = False
        y.parent = x

        if y.left == y.right:
            y.left._set_siblings(None, None)
        else:
            y.left.right = y.right
            y.right.left = y.left

        x.add_child(y)

    @_inherit_doc(MinHeap)
    def insert(self, key, value, allow_increase=False):

        if allow_increase:
            raise NotImplementedError

        # Create a FHeap item for our key value
        item = self._FHeapItem(key, value)

        # If the heap is empty, this is the new item of the heap
        if self.min_item is None:
            self.min_item = item

        # If the heap has a min item, add the new item to the right of it and
        # compare against the minimum
        else:
            self.min_item.add_right(item)
            if value < self.min_item.value:
                self.min_item = item

        self.count += 1
        return True

