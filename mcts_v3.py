from input_data import *

"""
第三版本
给定一个初始机翼，在此基础上的变化翼型。
翼型数组为由上到下。
"""


class Wing:

    def __init__(self, a, precision=11):
        self.precision = precision
        self.middle = (precision + 1) // 2
        self.airfoil = self.middle * a
        self.fi = a.shape[0] * 2 - 3
        self.index = np.argwhere(self.airfoil != 0)
        self.length = 1

    def clone(self):
        w = Wing(self.airfoil.copy())
        w.length = self.length
        return w

    def draw(self, p):
        ind = self.index[self.length]
        if 1 < p < self.precision:
            self.airfoil[tuple(ind)] = p
        else:
            self.airfoil[tuple(ind)] = 0
            ind[1] += -1 if p == 1 else 1
            self.airfoil[tuple(ind)] = self.middle
        self.length += 1

    def finish(self):
        return False if self.length < self.fi else True


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
        return sorted(self.children,
                      key=lambda c: c.Q + c.P * np.sqrt(2 * np.log(self.visit)) / (1 + c.visit))[-1]

    def update(self):
        self.visit += 1
        self.W = self.V + sum(map(lambda c: c.W, self.children))
        self.Q = self.W / (self.children_number + 1)
        if self.parent:
            self.parent.children_number += self.new_children


def uct(root_wing, iteration, pred, v, x, se, t):
    root_node = Node()

    for i in range(iteration):
        node = root_node
        wing = root_wing.clone()

        while node.children:
            node = node.select_child()
            wing.draw(node.index)

        if not wing.finish():
            node.pred = se.run(pred, {x: wing.airfoil}).reshape(-1)
            node.value = se.run(v, {x: wing.airfoil}).reshape(-1)
            node.add_child()

        while node:
            node.update()
            node = node.parent

    pi = np.array(list(map(lambda c: c.visit, root_node.children)))
    pi = np.power(pi, 1 / t)
    pi = pi / np.sum(pi)
    return pi, np.argmax(pi)


if __name__ == '__main__':
    airf = naca0012(points(21))
    print(airf)
