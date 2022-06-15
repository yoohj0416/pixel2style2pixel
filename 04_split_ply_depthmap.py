import vedo
from pathlib import Path
import open3d as o3d
import glob
import os
from PIL import Image
import numpy as np


def main():
    plyfiles = glob.glob('/home/malab4/dataArchive/DCPR-GAN-Data/5-out_boundary/*.ply')
    SAVE_DIR = '/home/malab4/dataArchive/DCPR-GAN-Data/5-out_boundary_depth'

    for pfile in plyfiles:
        pname = os.path.basename(pfile).replace('.ply', '.png')
        print(pfile, '-' * 50)
        print(pname)

        vedo_mesh = vedo.load(pfile)

        splited = vedo_mesh.split()
        cells_list = []
        for splited_mesh in splited:
            cells_list.append(splited_mesh.NCells())
        vedo_mesh_splited = splited[cells_list.index(max(cells_list))]

        vedo_pcd = vedo_mesh_splited.points()
        vedo_cells = vedo_mesh_splited.cells()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vedo_pcd)
        mesh.triangles = o3d.utility.Vector3iVector(vedo_cells)

        mesh.compute_vertex_normals()

        pcd = mesh.sample_points_uniformly(number_of_points=1600000)

        pts = np.asarray(pcd.points)
        print(f'num. of points: {len(pts)}')
        print(f'shape of points: {pts.shape}')

        xscale = 200

        xmax = np.max(pts[:, 0])
        xmin = np.min(pts[:, 0])
        ymax = np.max(pts[:, 1])
        ymin = np.min(pts[:, 1])
        zmin = np.min(pts[:, 2]) - 1
        zmax = zmin + 6

        yscale = int(xscale * (ymax - ymin) / (xmax - xmin))

        print(f'xscale {xscale}, yscale {yscale}')

        depthmap = np.zeros((yscale + 2, xscale + 2), dtype=np.uint8)

        for pt in pts:
            x, y, z = pt
            x = int(xscale * (x - xmin) / (xmax - xmin))
            y = int(yscale * (y - ymin) / (ymax - ymin))
            if (z - zmin) / (zmax - zmin) > 1:
                continue
            if depthmap[y][x] < 255 - int(255 * (z - zmin) / (zmax - zmin)):
                depthmap[y][x] = int(255 * ((z - zmin) / (zmax - zmin)) ** 2)

        depthimg = Image.fromarray(depthmap)
        depthimg.save(os.path.join(SAVE_DIR, pname))


if __name__ == '__main__':
    main()