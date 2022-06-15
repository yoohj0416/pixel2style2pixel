import cv2
import numpy as np
import open3d as o3d
import glob
import os
from PIL import Image


def main():
    # plyfiles = glob.glob('/home/malab4/dataArchive/DCPR-GAN-Data/1-Opposing_teeth/*.ply')
    # plyfiles = glob.glob('/home/malab4/dataArchive/DCPR-GAN-Data/2-Object/*.ply')
    # plyfiles = glob.glob('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation/*.ply')
    plyfiles = glob.glob('/home/malab4/dataArchive/DCPR-GAN-Data/4-exception/*.ply')
    # SAVE_DIR = '/home/malab4/dataArchive/DCPR-GAN-Data/1-Opposing_teeth_depth'
    # SAVE_DIR = '/home/malab4/dataArchive/DCPR-GAN-Data/2-Object_depth'
    # SAVE_DIR = '/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation_depth'
    # SAVE_DIR = '/home/malab4/dataArchive/DCPR-GAN-Data/4-exception_depth'
    SAVE_DIR = '/home/malab4/dataArchive/DCPR-GAN-Data/4-exception_depth__'

    if 'opposing' in plyfiles[0].lower():
        is_opposing = True
    else:
        is_opposing = False

    for pfile in plyfiles:
        pname = os.path.basename(pfile).replace('.ply', '.png')
        print(pfile, '-' * 50)
        print(pname)

        mesh = o3d.io.read_triangle_mesh(pfile)
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
        if is_opposing:
            zmin = np.min(pts[:, 2]) - 1
            zmax = zmin + 6
        else:
            zmax = np.max(pts[:, 2]) + 1
            zmin = zmax - 6

        yscale = int(xscale * (ymax - ymin) / (xmax - xmin))

        print(f'xscale {xscale}, yscale {yscale}')

        depthmap = np.zeros((yscale + 2, xscale + 2), dtype=np.uint8)

        for pt in pts:
            x, y, z = pt
            x = int(xscale * (x - xmin) / (xmax - xmin))
            y = int(yscale * (y - ymin) / (ymax - ymin))
            if is_opposing:
                if (z - zmin) / (zmax - zmin) > 1:
                    continue
                if depthmap[y][x] < 255 - int(255 * (z - zmin) / (zmax - zmin)):
                    depthmap[y][x] = int(255 * ((z - zmin) / (zmax - zmin)) ** 2)
            else:
                if depthmap[y][x] < int(255 * (z - zmin) / (zmax - zmin)):
                    depthmap[y][x] = int(255 * ((z - zmin) / (zmax - zmin)) ** 2)
                    # when opposing exception
                    # depthmap[y][x] = 255 - int(255 * np.sqrt((z - zmin) / (zmax - zmin)))

        # cv2.imshow('depthmap', depthmap)
        # cv2.waitKey()
        # exit(0)
        #
        depthimg = Image.fromarray(depthmap)
        depthimg.save(os.path.join(SAVE_DIR, pname))


if __name__ == '__main__':
    main()