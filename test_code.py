import open3d as o3d
import numpy as np
from PIL import Image


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

    # true = 0
    # false = 0
    # for i in range(100000):
    #     if np.random.randint(2):
    #         true += 1
    #     else:
    #         false += 1
    #
    # print(true, false)

    img1 = Image.open("/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/train/3-Preparation/data0001.png")
    img1 = img1.convert('L')
    # img1 = img1.convert('RGB')
    img2 = Image.open("/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/train/1-Opposing_teeth/data0001.png")
    img2 = img2.convert('L')
    # img2 = img2.convert('RGB')

    zero_im = np.expand_dims(np.zeros(img2.size, dtype=np.uint8), axis=2)
    img1 = np.expand_dims(np.array(img1), axis=2)
    img2 = np.expand_dims(np.array(img2), axis=2)

    print(img1. shape)
    print(img2. shape)
    print(zero_im. shape)

    # to_im = np.concatenate((img1, img2, zero_im), axis=2)
    to_im = Image.fromarray(np.concatenate((img1, img2, zero_im), axis=2))
    # print(to_im.shape)
    print(to_im.size)
    print(to_im)
    # print(len(to_im.mode))
    # print(np.max(to_im))



if __name__ == '__main__':
    main()