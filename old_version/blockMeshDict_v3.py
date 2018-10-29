import numpy as np


def cal_vertices(airf):
    n = airf.shape[0]
    x = np.argwhere(airf.T == 1).astype(np.float)
    x[:, 1] -= x[0, 1]
    x[:, 0] /= n - 1
    x[:, 1] /= (n - 1) * 2
    return x


def array_sort(v):
    v = v[v[:, 0] > 0]
    down = v[v[:, 1] < 0.0001]
    down = down[np.argsort(down[:, 0])][::-1]
    up = v[v[:, 1] > 0.0001]
    up = up[np.argsort(up[:, 0])]

    return np.concatenate((down, up))


def tran(v):
    o = np.array([0.3, 0, 0])
    if v[0] <= o[0]:
        return 2 * (v - o) / np.linalg.norm(v - o) + o
    elif v[2] < 0.0001:
        return np.array([v[0], 0, -2])
    else:
        return np.array([v[0], 0, 2])


def write_vertices(file, v):
    file.write('vertices\n'
               '(\n')
    for ver in v:
        file.write('   (   %f      %f      %f  )\n' % (ver[0], ver[1], ver[2]))
        file.write('   (   %f      %f      %f  )\n' % (ver[0], ver[1] + 0.2, ver[2]))
    file.write(');\n\n')


def write_edges(file, m):
    n = (m - 1) * 2
    file.write('edges\n'
               '(\n'
               '    arc %d %d (%f %f %f)\n'
               '    arc %d %d (%f %f %f)\n'
               '    arc %d %d (%f %f %f)\n'
               '    arc %d %d (%f %f %f)\n'
               ');\n\n'
               % (m - 1, m + 1, 0, 0, 0,
                  m, m + 2, 0, 0.2, 0,
                  m + n - 1, m + n + 1, -1.7, 0, 0,
                  m + n, m + n + 2, -1.7, 0.2, 0))


def write_blocks(file, m, b):
    bo = [0, 4, 6, 2, 1, 5, 7, 3]
    file.write('blocks\n'
               '(\n')
    for i in range(m):
        tup = tuple(b[i:i + 2].reshape(-1)[bo])
        if i == 0 or i == m - 1:
            file.write('    hex (%d %d %d %d %d %d %d %d) (40 1 80) simpleGrading (1 1 10)\n' % tup)
        elif i == m // 2:
            file.write('    hex (%d %d %d %d %d %d %d %d) (4 1 80) simpleGrading (1 1 10)\n' % tup)
        else:
            file.write('    hex (%d %d %d %d %d %d %d %d) (1 1 80) simpleGrading (1 1 10)\n' % tup)
    file.write(');\n\n')


def write_boundary(file, m, b):
    wo = [0, 2, 6, 4]
    io = [1, 5, 7, 3]
    fo = [0, 4, 5, 1]
    bo = [2, 3, 7, 6]
    file.write('boundary\n'
               '(\n'
               '    wing\n'
               '    {\n'
               '        type wall;\n'
               '        faces\n'
               '        (\n')
    for i in range(m - 2):
        tup = tuple(b[i + 1: i + 3].reshape(-1)[wo])
        file.write('            (%d %d %d %d)\n' % tup)
    file.write('        );\n'
               '    }\n\n')

    file.write('    inlet\n'
               '    {\n'
               '        type patch;\n'
               '        inGroups 1(freestream);\n'
               '        faces\n'
               '        (\n')
    for i in range(m):
        tup = tuple(b[i: i + 2].reshape(-1)[io])
        file.write('            (%d %d %d %d)\n' % tup)
    file.write('        );\n'
               '    }\n\n'
               '    outlet\n'
               '    {\n'
               '        type patch;\n'
               '        inGroups 1(freestream);\n'
               '        faces\n'
               '        (\n')

    tup1 = tuple(b[0].reshape(-1)[[0, 1, 3, 2]])
    tup2 = tuple(b[-1].reshape(-1)[[0, 2, 3, 1]])
    file.write('            (%d %d %d %d)\n' % tup1)
    file.write('            (%d %d %d %d)\n' % tup2)
    file.write('        );\n'
               '    }\n\n'
               '    front\n'
               '    {\n'
               '        type empty;\n'
               '        faces\n'
               '        (\n')
    for i in range(m):
        tup = tuple(b[i:i + 2].reshape(-1)[fo])
        file.write('            (%d %d %d %d)\n' % tup)
    file.write('        );\n'
               '    }\n\n'
               '    back\n'
               '    {\n'
               '        type empty;\n'
               '        faces\n'
               '        (\n')
    for i in range(m):
        tup = tuple(b[i:i + 2].reshape(-1)[bo])
        file.write('            (%d %d %d %d)\n' % tup)
    file.write('        );\n'
               '    }\n'
               ');\n\n')


def write_dict(a):

    if a.shape[1] != 2:
        a = cal_vertices(a)

    v = array_sort(a)
    m1 = v.shape[0]
    m2 = m1 * 2 + 4
    blocks_num = m1 + 2
    points_num = m2 * 2

    v = np.insert(v, 0, [4, 0], axis=0)
    v = np.insert(v, 1, 0, axis=1)

    o = np.array([0.3, 0])
    n = v[v[:, 0] <= o[0]].shape[0]
    theta = np.linspace(0, np.pi, n)
    circle = np.insert(np.array([-np.sin(theta), -np.cos(theta)]).T * 2 + o, 1, 0, axis=1)

    bottom = v[(v[:, 0] > o[0]) * (v[:, 2] < 0.0001)]
    bottom[:, 2] = -2
    top = v[(v[:, 0] > o[0]) * (v[:, 2] > 0.0001)]
    top[:, 2] = 2

    vertices = np.concatenate((v, bottom, circle, top, [[1, 0, 2], [4, 0, 2]]), axis=0)

    blocks = np.arange(points_num)
    blocks = np.insert(blocks, m1 * 2 + 2, [2, 3, 0, 1]).reshape(2, -1).T.reshape(-1, 2, 2)

    with open('/home/xie/PycharmProjects/wing/wing_openFoam/system/blockMeshDict', 'w') as f:
        f.write('FoamFile\n'
                '{\n'
                '   version     4.1;\n'
                '   format      ascii;\n'
                '   class       dictionary;\n'
                '   object      blockMeshDict;\n'
                '}\n'
                '\n'
                '\n')
        write_vertices(f, vertices)
        write_edges(f, blocks_num)
        write_blocks(f, blocks_num, blocks)
        write_boundary(f, blocks_num, blocks)
        f.write('mergePatchPairs\n'
                '(\n'
                ');\n\n')


airfoil_pre = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape([11, 11])


if __name__ == '__main__':

    x = np.argwhere(airfoil_pre.T == 1).astype(np.float)
    x[:, 1] -= x[0, 1]
    x /= np.array([10, 30])
    x = np.array([1, 0, 0.1, -0.1, 0, 0, 0.1, 0.1, 0.3, 0.1, 0.3, -0.1]).reshape(-1, 2)
    write_dict(x)
