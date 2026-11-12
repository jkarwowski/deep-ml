from collections import deque
from collections import Counter

def dfs(node: "Value", visited: "Counter[Value]") -> "set[Value]":
    nodes = set()
    for c in node._prev:
        visited[c] += 1
        if visited[c] == 1:
            dfs(c, visited)
    return visited


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op


    def __repr__(self):
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)
        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    def __add__(self, other):
        v = Value(data = self.data + other.data, _children=(self, other), _op="+")
        def _backward():
            self.grad += v.grad
            other.grad += v.grad
        v._backward = _backward
        return v

    def __mul__(self, other):
        # Implement multiplication here
        v = Value(data = self.data * other.data, _children=(self, other), _op="*")
        def _backward():
            self.grad += other.data * v.grad
            other.grad += self.data * v.grad
        v._backward = _backward
        return v

    def relu(self):
        # Implement ReLU here
        v = Value(data = int(self.data > 0) * self.data, _children=(self, ), _op="relu")
        def _backward():
            self.grad += int(self.data > 0) * v.grad
        v._backward = _backward
        return v

    def backward(self):
        # Implement backward pass here
        # first build a topologically sorted graph of _children
        sorted_children = []
        q = deque()
        all_parents = Counter()
        dfs(self, all_parents)
        q.append(self)

        # zero gradients
        for p in all_parents.keys():
            p.grad = 0

        while q:
            c = q.popleft()
            sorted_children.append(c)

            for kid in c._prev:
                all_parents[kid] -= 1
                if all_parents[kid] == 0:
                    q.append(kid)
            # print([(c.data, c.grad) for c in sorted_children])
        assert len(sorted_children) == len(all_parents.keys()) + 1, str((sorted_children, all_parents))
        
        # print("Grad")
        self.grad = 1
        for c in sorted_children:
            # print(c.data, [(c.data, c.grad) for c in sorted_children])
            c._backward()
