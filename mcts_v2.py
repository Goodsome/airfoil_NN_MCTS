import numpy as np

"""
mcts第二版本
神经网络输出数组V，储存子节点v_i

"""


def index_point(index, n):
    assert index < n * (n - 1) / 2
    point = np.zeros(n)
    i = 0
    while index >= n - 1:
        index -= (n - 1)
        n -= 1
        i += 1
    point[[i, i + index + 1]] = 1
    return point


def point_index(point, n):
    a = np.argwhere(point == 1).reshape(-1)
    i = a[0]
    index = a[-1] - i - 1

    for _ in range(i):
        index += n - 1
        n -= 1

    return index


class Wing:

    def __init__(self, airfoil):
        self.airfoil = airfoil
        self.n = airfoil.shape[1]
        self.length = 1

    def clone(self):
        w = Wing(self.airfoil.copy())
        w.length = self.length
        return w

    def draw(self, point):
        self.airfoil[:, self.length] = point
        self.length += 1

    def finish(self):
        return False if self.length < self.n - 1 else True


class Node:

    def __init__(self, parent=None, index=None, p=0, v=0):
        self.parent = parent
        self.index = index
        self.P = p
        self.V = v
        self.pred = None
        self.value = None
        self.visit = 0
        self.W = v
        self.Q = v
        self.children = []
        self.new_children = 0
        self.children_number = 0

    def add_child(self):
        for i in range(self.pred.shape[0]):
            self.children.append(Node(parent=self, index=i, p=self.pred[i], v=self.value[i]))
        self.new_children = self.pred.shape[0]
        self.children_number += self.new_children

    def select_child(self):
        return sorted(self.children, key=lambda c: c.P / (1 + c.visit) + c.Q)[-1]

    def update(self):
        self.visit += 1
        self.W = self.V + sum(map(lambda c: c.W, self.children))
        self.Q = self.W / (self.children_number + 1)
        if self.parent:
            self.parent.children_number += self.new_children


def uct(root_wing, iteration, pred, v, x, se):
    root_node = Node()

    for i in range(iteration):
        node = root_node
        wing = root_wing.clone()

        while node.children:
            node = node.select_child()
            point = index_point(node.index, wing.n)
            wing.draw(point)

        if not wing.finish():
            node.pred = se.run(pred, {x:wing.airfoil}).reshape(-1)
            node.value = se.run(v , {x:wing.airfoil}).reshape(-1)
            node.add_child()

        while node:
            node.update()
            node = node.parent

    pi = np.array(list(map(lambda c: c.visit, root_node.children)))
    pi = pi / np.sum(pi)
    return pi, np.argmax(pi)

