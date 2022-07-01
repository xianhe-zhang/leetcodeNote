# 1603. Design Parking System
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.big = big
        self.medium = medium
        self.small = small
        

    def addCar(self, carType: int) -> bool:
        if carType == 1:
            self.big -= 1
            return self.big >= 0
        elif carType == 2:
            self.medium -= 1
            return self.medium >= 0
        else:
            self.small -= 1
            return self.small >= 0



# 705. Design HashSet
class MyHashSet(object):

    def __init__(self):
        self.keyRange = 769
        self.bucketArray = [Bucket() for i in range(self.keyRange)]

    def _hash(self, key):
        return key % self.keyRange

    def add(self, key):
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].insert(key)

    def remove(self, key):
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].delete(key)

    def contains(self, key):
        bucketIndex = self._hash(key)
        return self.bucketArray[bucketIndex].exists(key)


class Node:
    def __init__(self, value, nextNode=None):
        self.value = value
        self.next = nextNode

class Bucket:
    def __init__(self):
        # a pseudo head/ Dummy head
        self.head = Node(0)

    def insert(self, newValue):# 插入到头部
        # if not existed, add the new element to the head.
        if not self.exists(newValue):
            newNode = Node(newValue, self.head.next)
            # set the new head.
            self.head.next = newNode

    def delete(self, value):
        prev = self.head
        curr = self.head.next
        while curr is not None:
            if curr.value == value:
                # remove the current node
                prev.next = curr.next
                return
            prev = curr
            curr = curr.next

    def exists(self, value):
        curr = self.head.next
        while curr is not None:
            if curr.value == value:
                # value existed already, do nothing
                return True
            curr = curr.next
        return False


# 706. Design HashMap
class Bucket:
    def __init__(self):
        self.bucket = []

    def get(self, key):
        for (k, v) in self.bucket:
            if k == key:
                return v
        return -1

    def update(self, key, value):
        found = False
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key, value)
                found = True
                break

        if not found:
            self.bucket.append((key, value))

    def remove(self, key):
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                del self.bucket[i]

class MyHashMap(object):
    def __init__(self):
        # better to be a prime number, less collision
        self.key_space = 2069
        self.hash_table = [Bucket() for i in range(self.key_space)]


    def put(self, key, value):
        hash_key = key % self.key_space
        self.hash_table[hash_key].update(key, value)


    def get(self, key):
        hash_key = key % self.key_space
        return self.hash_table[hash_key].get(key)


    def remove(self, key):
        hash_key = key % self.key_space
        self.hash_table[hash_key].remove(key)

        
# 703. Kth Largest Element in a Stream
class KthLargest
    def __init__(self, k: int, nums: List[int]):
        self.nums = nums
        self.k = k

    def add(self, val: int) -> int:
        self.nums.append(val)
        
        self.nums.sort()
        return self.nums[-self.k]

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]



# 146. LRU Cache
from collections import OrderedDict
class LRUCache(OrderedDict):

    def __init__(self, capacity: int):
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self:
            return -1
        self.move_to_end(key)
        return self[key]
        

    def put(self, key: int, value: int) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last = False)



# 146. LRU Cache
# 想要同时实现排序与查找，不用dict的情况下，可以用双向链表
class DLinkedNode(): 
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None
            
class LRUCache():
    def _add_node(self, node):
        node.prev = self.head
        node.next = self.head.next

        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        prev = node.prev
        new = node.next

        prev.next = new
        new.prev = prev

    def _move_to_head(self, node):
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self):
        res = self.tail.prev
        self._remove_node(res)
        return res

    def __init__(self, capacity):

        self.cache = {}
        self.size = 0
        self.capacity = capacity
        self.head, self.tail = DLinkedNode(), DLinkedNode()

        self.head.next = self.tail
        self.tail.prev = self.head
        

    def get(self, key):
        node = self.cache.get(key, None)
        if not node:
            return -1

        # move the accessed node to the head;
        self._move_to_head(node)

        return node.value

    def put(self, key, value):
        node = self.cache.get(key)

        if not node: 
            newNode = DLinkedNode()
            newNode.key = key
            newNode.value = value

            self.cache[key] = newNode
            self._add_node(newNode)

            self.size += 1

            if self.size > self.capacity:
                # pop the tail
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1
        else:
            # update the value.
            node.value = value
            self._move_to_head(node)



# 341. Flatten Nested List Iterator
# 这种解法主要是用来学习一些API和stack的思想。
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = list(reversed(nestedList)) # 为什么reverse，因为我们用的是pop()，效率比pop(0)要高
        
    def next(self) -> int:
        self.make_stack_top_an_integer()
        return self.stack.pop().getInteger() # getInteger我之前么有用到过
    
        
    def hasNext(self) -> bool:
        self.make_stack_top_an_integer()
        return len(self.stack) > 0
        
    # 重点在于这个
    def make_stack_top_an_integer(self):
        while self.stack and not self.stack[-1].isInteger():
            self.stack.extend(reversed(self.stack.pop().getList()))

# 直接在初始化的时候通过recursion的方式falttern掉nestedList，通过一个信号量判断flag
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        def flatten_list(nested_list):
            for nested_integer in nested_list:
                if nested_integer.isInteger():
                    self._integers.append(nested_integer.getInteger())
                else:
                    flatten_list(nested_integer.getList()) 
        self._integers = []
        self._position = -1 # Pointer to previous returned.
        flatten_list(nestedList)
    
    def next(self) -> int:
        self._position += 1
        return self._integers[self._position]
        
    def hasNext(self) -> bool:
        return self._position + 1 < len(self._integers)


# 208. Implement Trie (Prefix Tree)

class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    

    def insert(self, word: str) -> None:
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word: str) -> bool:
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True
   


# 173. Binary Search Tree Iterator
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.nodes_sorted = []
        self.index = -1         # 这里之所以是-1，因为要处理一些index range的问题。
        self._inorder(root)
    
    def _inorder(self, root):
        if not root:
            return 
        self._inorder(root.left)
        self.nodes_sorted.append(root.val)
        self._inorder(root.right)

    def next(self) -> int:
        self.index += 1
        return self.nodes_sorted[self.index]
        
    def hasNext(self) -> bool:
        return self.index + 1 < len(self.nodes_sorted)
        

# 622. Design Circular Queue
class MyCircularQueue:

    def __init__(self, k: int):
        self.queue = []
        self.capacity = k
        self.index = 0  # 这里的index更像是一个信号量的存在，不需要参与计算，因此直接从0开始就可以。
        self.count = 0
        

    def enQueue(self, value: int) -> bool:
        if self.index == self.capacity:
            return False
        self.queue.append(value)
        self.index += 1
        return True

    def deQueue(self) -> bool:
        if self.index == 0:
            return False
        self.queue.pop(0)
        self.index -= 1
        return True
        

    def Front(self) -> int:
        return self.queue[0] if self.queue else -1

    def Rear(self) -> int:
        print(self.queue)
        return self.queue[-1] if self.queue else -1
        

    def isEmpty(self) -> bool:
        return self.index == 0

    def isFull(self) -> bool:
        return self.index == self.capacity


# 380. Insert Delete GetRandom O(1)
from random import choice
class RandomizedSet():
    def __init__(self):
        self.dict = {}
        self.list = []

        
    def insert(self, val: int) -> bool:
        if val in self.dict:
            return False
        self.dict[val] = len(self.list) # 这个就是为了给val上index1
        self.list.append(val)
        return True

    # dict中存放着index和value，我们把最后一位数往list中替换掉，然后pop最后一位
    def remove(self, val: int) -> bool:
        if val in self.dict:
            last_element, idx = self.list[-1], self.dict[val]
            self.list[idx], self.dict[last_element] = last_element, idx
            
            self.list.pop()  #更新list
            del self.dict[val] #更新dict
            return True
        return False

    def getRandom(self) -> int:
        return choice(self.list)