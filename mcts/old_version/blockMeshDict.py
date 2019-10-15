import numpy as np
from blockMeshDict_v2 import airfoil_pre


def cal_vertices(airf):
    co = np.argwhere(airf.T == 1)
    co[:, 1] -= co[0, 1]
    co = co.astype(float)
    co /= np.max(np.abs(co), axis=0)
    co[:, 1] *= 0.06

    m = co.shape[0]
    m1 = m * 2 + 8
    m2 = m // 2 + 3

    vertices = np.zeros([m1, 3])
    vertices[:m2, 0] = np.concatenate(([-2], co[::2, 0], [1, 4]))
    vertices[:m2, 2] = 2
    vertices[m2:2 * m2, 0] = vertices[:m2, 0]
    vertices[m2:2 * m2, 2] = np.concatenate(([0], co[::2, 1], [0, 0]))
    vertices[2 * m2:-m2, 0] = co[::2, 0][1:]
    vertices[2 * m2:-m2, 2] = co[1::2, 1][:-1]
    vertices[-m2:, 0] = vertices[:m2, 0]
    vertices[-m2:, 2] = -2

    return vertices, m


def write_vertices(file, v):
    file.write('vertices\n'
               '(\n')
    for ver in v:
        file.write('   (   %f      %f      %f  )\n' % (ver[0], ver[1], ver[2]))
        file.write('   (   %f      %f      %f  )\n' % (ver[0], ver[1] + 0.2, ver[2]))
    file.write(');\n\n')


def write_edges(file):
    file.write('edges\n'
               '(\n'
               ');\n\n')


def write_blocks(file, m2, point, block_order):
    file.write('blocks\n'
               '(\n')
    for i in range(m2 - 1):
        tup1 = tuple(point[:, :2, i:i + 2].reshape(-1)[block_order])
        tup2 = tuple(point[:, 2:4, i:i + 2].reshape(-1)[block_order])
        if i == 0:
            file.write('    hex (%d %d %d %d %d %d %d %d) (50 1 100) simpleGrading (0.1 1 10)\n' % tup1)
            file.write('    hex (%d %d %d %d %d %d %d %d) (50 1 100) simpleGrading (0.1 1 0.1)\n' % tup2)
        elif i == m2 - 2:
            file.write('    hex (%d %d %d %d %d %d %d %d) (50 1 100) simpleGrading (10 1 10)\n' % tup1)
            file.write('    hex (%d %d %d %d %d %d %d %d) (50 1 100) simpleGrading (10 1 0.1)\n' % tup2)
        else:
            file.write('    hex (%d %d %d %d %d %d %d %d) (6 1 100) simpleGrading (1 1 10)\n' % tup1)
            file.write('    hex (%d %d %d %d %d %d %d %d) (6 1 100) simpleGrading (1 1 0.1)\n' % tup2)
    file.write(');\n\n')


def write_boundary(file, m2, point, order1, order2):
    file.write('boundary\n'
               '(\n'
               '    wing\n'
               '    {\n'
               '        type wall;\n'
               '        faces\n'
               '        (\n')
    for i in range(m2 - 3):
        downface_tup = tuple(point[:, 1, i + 1: i + 3].reshape(-1)[order2])
        upface_tup = tuple(point[:, 2, i + 1: i + 3].reshape(-1)[order1])
        file.write('            (%d %d %d %d)\n'
                   '            (%d %d %d %d)\n'
                   % (upface_tup + downface_tup))
    file.write('        );\n'
               '    }\n\n')

    file.write('    inlet\n'
               '    {\n'
               '        type patch;\n'
               '        inGroups 1(freestream);\n'
               '        faces\n'
               '        (\n')
    for i in range(2):
        tup = tuple(point[:, i * 2:i * 2 + 2, 0].reshape(-1)[order2])
        file.write('            (%d %d %d %d)\n' % tup)

    for i in range(m2 - 1):
        tup = tuple(point[:, 3, i: i + 2].reshape(-1)[order2])
        file.write('            (%d %d %d %d)\n' % tup)
    file.write('        );\n'
               '    }\n\n'
               '    outlet\n'
               '    {\n'
               '        type patch;\n'
               '        inGroups 1(freestream);\n'
               '        faces\n'
               '        (\n')

    for i in range(2):
        tup = tuple(point[:, i * 2:i * 2 + 2, -1].reshape(-1)[order1])
        file.write('            (%d %d %d %d)\n' % tup)

    for i in range(m2 - 1):
        tup = tuple(point[:, 0, i: i + 2].reshape(-1)[order1])
        file.write('            (%d %d %d %d)\n' % tup)
    file.write('        );\n'
               '    }\n\n'
               '    front\n'
               '    {\n'
               '        type empty;\n'
               '        faces\n'
               '        (\n')
    for i in range(m2 - 1):
        tup1 = tuple(point[0, :2, i:i + 2].reshape(-1)[order2])
        tup2 = tuple(point[0, 2:4, i:i + 2].reshape(-1)[order2])
        file.write('            (%d %d %d %d)\n'
                   '            (%d %d %d %d)\n' % (tup1 + tup2))
    file.write('        );\n'
               '    }\n\n'
               '    back\n'
               '    {\n'
               '        type empty;\n'
               '        faces\n'
               '        (\n')
    for i in range(m2 - 1):
        tup1 = tuple(point[1, :2, i:i + 2].reshape(-1)[order1])
        tup2 = tuple(point[1, 2:4, i:i + 2].reshape(-1)[order1])
        file.write('            (%d %d %d %d)\n'
                   '            (%d %d %d %d)\n' % (tup1 + tup2))

    file.write('        );\n'
               '    }\n'
               ');\n\n')


def write_block_mesh_dict(v, m):

    m1 = m * 2 + 8
    m2 = m // 2 + 3

    point = np.arange(2 * m1).reshape(-1, 2).T
    point = np.insert(point, 2 * m2, [[2 * m2, 2 * m2 + 1], [2 * m2 + 2, 2 * m2 + 3]], axis=1)
    point = np.insert(point, -m2, [[4 * m2 - 4, 4 * m2 - 3], [4 * m2 - 2, 4 * m2 - 1]], axis=1).reshape(2, 4, -1)

    block_order = [2, 3, 7, 6, 0, 1, 5, 4]
    order1 = [0, 1, 3, 2]
    order2 = [0, 2, 3, 1]

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
        write_vertices(f, v)
        write_edges(f)
        write_blocks(f, m2, point, block_order)
        write_boundary(f, m2, point, order1, order2)
        f.write('mergePatchPairs\n'
                '(\n'
                ');\n\n')


if __name__ == '__main__':
    ver, num = cal_vertices(airfoil_pre)
    write_block_mesh_dict(ver, num)
