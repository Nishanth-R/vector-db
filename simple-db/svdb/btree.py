"""
B-tree implementation for efficient indexing and querying by primary key.
"""
import pickle
import os
from typing import Any, Optional, List, Tuple
from .errors import LoadingException


class BTreeNode:
    """Represents a single node in the B-tree."""
    
    def __init__(self, is_leaf: bool = True, max_keys: int = 3):
        self.is_leaf = is_leaf
        self.keys: List[Any] = []
        self.values: List[Any] = []
        self.children: List['BTreeNode'] = []
        self.max_keys = max_keys
    
    def search(self, key: Any) -> Optional[Any]:
        """Search for a key in this node."""
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1
        
        if i < len(self.keys) and self.keys[i] == key:
            return self.values[i]
        
        if self.is_leaf:
            return None
        
        return self.children[i].search(key)
    
    def insert_non_full(self, key: Any, value: Any):
        """Insert a key-value pair into a non-full node."""
        i = len(self.keys) - 1
        
        if self.is_leaf:
            # Insert into leaf node
            self.keys.append(0)
            self.values.append(0)
            
            while i >= 0 and self.keys[i] > key:
                self.keys[i + 1] = self.keys[i]
                self.values[i + 1] = self.values[i]
                i -= 1
            
            self.keys[i + 1] = key
            self.values[i + 1] = value
        else:
            # Find child to insert into
            while i >= 0 and self.keys[i] > key:
                i -= 1
            
            i += 1
            if len(self.children[i].keys) == self.max_keys:
                self.split_child(i, self.children[i])
                if self.keys[i] < key:
                    i += 1
            
            self.children[i].insert_non_full(key, value)
    
    def split_child(self, index: int, child: 'BTreeNode'):
        """Split a full child node."""
        new_child = BTreeNode(is_leaf=child.is_leaf, max_keys=self.max_keys)
        
        # Move the last (max_keys//2) keys and values to new_child
        mid = self.max_keys // 2
        new_child.keys = child.keys[mid + 1:]
        new_child.values = child.values[mid + 1:]
        child.keys = child.keys[:mid + 1]
        child.values = child.values[:mid + 1]
        
        if not child.is_leaf:
            new_child.children = child.children[mid + 1:]
            child.children = child.children[:mid + 1]
        
        # Insert the middle key into this node
        self.children.insert(index + 1, new_child)
        self.keys.insert(index, child.keys[mid])
        self.values.insert(index, child.values[mid])
        child.keys = child.keys[:mid]
        child.values = child.values[:mid]
    
    def delete(self, key: Any) -> bool:
        """Delete a key from the B-tree. Returns True if key was found and deleted."""
        i = 0
        while i < len(self.keys) and self.keys[i] < key:
            i += 1
        
        if i < len(self.keys) and self.keys[i] == key:
            if self.is_leaf:
                self.keys.pop(i)
                self.values.pop(i)
                return True
            else:
                # Key is in internal node
                if len(self.children[i].keys) >= (self.max_keys + 1) // 2:
                    pred = self._get_predecessor(i)
                    self.keys[i] = pred[0]
                    self.values[i] = pred[1]
                    return self.children[i].delete(pred[0])
                elif len(self.children[i + 1].keys) >= (self.max_keys + 1) // 2:
                    succ = self._get_successor(i)
                    self.keys[i] = succ[0]
                    self.values[i] = succ[1]
                    return self.children[i + 1].delete(succ[0])
                else:
                    self._merge(i)
                    return self.children[i].delete(key)
        
        if self.is_leaf:
            return False
        
        flag = (i == len(self.keys))
        if len(self.children[i].keys) < (self.max_keys + 1) // 2:
            self._fill(i)
        
        if flag and i > len(self.keys):
            return self.children[i - 1].delete(key)
        
        return self.children[i].delete(key)
    
    def _get_predecessor(self, index: int) -> Tuple[Any, Any]:
        """Get the predecessor key-value pair."""
        current = self.children[index]
        while not current.is_leaf:
            current = current.children[-1]
        return current.keys[-1], current.values[-1]
    
    def _get_successor(self, index: int) -> Tuple[Any, Any]:
        """Get the successor key-value pair."""
        current = self.children[index + 1]
        while not current.is_leaf:
            current = current.children[0]
        return current.keys[0], current.values[0]
    
    def _fill(self, index: int):
        """Fill a child node that has less than minimum keys."""
        if index != 0 and len(self.children[index - 1].keys) >= (self.max_keys + 1) // 2:
            self._borrow_from_prev(index)
        elif index != len(self.keys) and len(self.children[index + 1].keys) >= (self.max_keys + 1) // 2:
            self._borrow_from_next(index)
        else:
            if index != len(self.keys):
                self._merge(index)
            else:
                self._merge(index - 1)
    
    def _borrow_from_prev(self, index: int):
        """Borrow a key from the previous sibling."""
        child = self.children[index]
        sibling = self.children[index - 1]
        
        child.keys.insert(0, self.keys[index - 1])
        child.values.insert(0, self.values[index - 1])
        
        if not child.is_leaf:
            child.children.insert(0, sibling.children.pop())
        
        self.keys[index - 1] = sibling.keys.pop()
        self.values[index - 1] = sibling.values.pop()
    
    def _borrow_from_next(self, index: int):
        """Borrow a key from the next sibling."""
        child = self.children[index]
        sibling = self.children[index + 1]
        
        child.keys.append(self.keys[index])
        child.values.append(self.values[index])
        
        if not child.is_leaf:
            child.children.append(sibling.children.pop(0))
        
        self.keys[index] = sibling.keys.pop(0)
        self.values[index] = sibling.values.pop(0)
    
    def _merge(self, index: int):
        """Merge child[index] with child[index+1]."""
        child = self.children[index]
        sibling = self.children[index + 1]
        
        child.keys.append(self.keys[index])
        child.values.append(self.values[index])
        
        child.keys.extend(sibling.keys)
        child.values.extend(sibling.values)
        
        if not child.is_leaf:
            child.children.extend(sibling.children)
        
        self.keys.pop(index)
        self.values.pop(index)
        self.children.pop(index + 1)


class BTree:
    """B-tree data structure for efficient key-value storage and retrieval."""
    
    def __init__(self, max_keys: int = 3, filename: Optional[str] = None):
        """
        Initialize a B-tree.
        
        Args:
            max_keys (int): Maximum number of keys per node (minimum degree = (max_keys+1)//2)
            filename (str, optional): File path to persist the B-tree
        """
        self.max_keys = max_keys
        self.root = BTreeNode(is_leaf=True, max_keys=max_keys)
        self.filename = filename
        if filename and os.path.exists(filename):
            self.load_from_disk()
    
    def search(self, key: Any) -> Optional[Any]:
        """Search for a value by key."""
        return self.root.search(key) if self.root else None
    
    def insert(self, key: Any, value: Any):
        """Insert a key-value pair into the B-tree."""
        if len(self.root.keys) == self.max_keys:
            new_root = BTreeNode(is_leaf=False, max_keys=self.max_keys)
            new_root.children.append(self.root)
            new_root.split_child(0, self.root)
            
            i = 0
            if new_root.keys[0] < key:
                i += 1
            new_root.children[i].insert_non_full(key, value)
            
            self.root = new_root
        else:
            self.root.insert_non_full(key, value)
        
        if self.filename:
            self.save_to_disk()
    
    def delete(self, key: Any) -> bool:
        """Delete a key from the B-tree."""
        if not self.root:
            return False
        
        result = self.root.delete(key)
        
        if not self.root.is_leaf and len(self.root.keys) == 0:
            self.root = self.root.children[0] if self.root.children else BTreeNode(is_leaf=True, max_keys=self.max_keys)
        
        if self.filename and result:
            self.save_to_disk()
        
        return result
    
    def save_to_disk(self):
        """Save the B-tree to disk."""
        if not self.filename:
            return
        
        try:
            with open(self.filename, 'wb') as file:
                pickle.dump(self.root, file)
        except Exception as err:
            raise LoadingException(f"Failed to save B-tree to disk: {str(err)}")
    
    def load_from_disk(self):
        """Load the B-tree from disk."""
        if not self.filename or not os.path.exists(self.filename):
            return
        
        try:
            with open(self.filename, 'rb') as file:
                self.root = pickle.load(file)
        except Exception as err:
            raise LoadingException(f"Failed to load B-tree from disk: {str(err)}")
    
    def get_all_keys(self) -> List[Any]:
        """Get all keys in the B-tree (in-order traversal)."""
        result = []
        self._in_order_traversal(self.root, result)
        return result
    
    def _in_order_traversal(self, node: BTreeNode, result: List[Any]):
        """Perform in-order traversal to collect all keys."""
        if node:
            i = 0
            for i in range(len(node.keys)):
                if not node.is_leaf:
                    self._in_order_traversal(node.children[i], result)
                result.append(node.keys[i])
            if not node.is_leaf:
                self._in_order_traversal(node.children[i + 1], result)

