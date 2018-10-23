import numpy as np
import time


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

    def __init__(self, parent=None, index=None, prob=None):
        self.parent = parent
        self.index = index
        self.P = prob
        self.pred = None
        self.value = 0.00001 * np.random.randn()
        self.visit = 0
        self.W = 0
        self.Q = 0.00001 * np.random.randn()
        self.children = []

    def add_child(self):
        for i, p in enumerate(self.pred.reshape(-1)):
            self.children.append(Node(parent=self, index=i, prob=p))

    def select_child(self):
        return sorted(self.children, key=lambda c: c.P / (1 + c.visit) + c.Q)[-1]

    def update(self):
        self.visit += 1
        self.W = self.value + sum(map(lambda c: c.W, self.children))
        self.Q = self.W / self.visit


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
            node.pred, node.value = se.run((pred, v), {x: wing.airfoil})
            node.add_child()

        while node:
            node.update()
            node = node.parent

    pi = np.array(list(map(lambda c: c.visit, root_node.children)))
    pi = pi / np.sum(pi)
    return pi, index_point(np.argmax(pi), root_wing.n)


if __name__ == '__main__':
    start = time.time()
    print(time.time() - start)
