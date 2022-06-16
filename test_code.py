import open3d as o3d
import numpy as np


def main():
    # # plyfile = '/home/malab4/dataArchive/DCPR-GAN-Data/2-Object/data0009.ply'
    # # plyfile = '/home/malab4/dataArchive/DCPR-GAN-Data/2-Object/data0007.ply'
    # # plyfile = '/home/malab4/dataArchive/DCPR-GAN-Data/2-Object/data0001.ply'
    # plyfile = '/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation/data0533.ply'
    #
    # mesh = o3d.io.read_triangle_mesh(plyfile)
    # pcd = o3d.io.read_point_cloud(plyfile)
    #
    # print(f"point cloud length: {len(pcd.points):,}")
    # print(f"mesh length: {len(mesh.triangles):,}")
    #
    # # o3d.visualization.draw_geometries([mesh, pcd])
    # o3d.visualization.draw_geometries([mesh])

    true = 0
    false = 0
    for i in range(100000):
        if np.random.randint(2):
            true += 1
        else:
            false += 1

    print(true, false)

if __name__ == '__main__':
    main()