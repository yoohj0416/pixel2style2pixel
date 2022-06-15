from pathlib import Path
import cv2
from depth_map_utils import fill_in_multiscale
import numpy as np
import vedo


def main():
    # depth_file = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation_depth/data0358.png')
    # depth_file = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation_depth/data0353.png')
    depth_file = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation_depth/data0349.png')
    # depth_file = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation_depth/data0348-1.png')
    #
    depth_img = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
    projected_depths = np.float32(depth_img / 256.0)

    final_depths, process_dict = fill_in_multiscale(projected_depths,
                                                    max_depth=256,
                                                    extrapolate=False,
                                                    blur_type='bilateral')
    #
    # cutted_depths = final_depths.copy()
    # for row_idx in range(depth_img.shape[0]):
    #     for col_idx in range(depth_img.shape[1]):
    #         left = False
    #         right = False
    #         # 왼쪽 픽셀에 0 이상의 값이 있는지 확인
    #         for i in range(0, col_idx):
    #             if depth_img[row_idx, i] > 0:
    #                 left = True
    #                 break
    #         # 오른쪽 픽셀에 0 이상의 값이 있는지 확인
    #         for i in range(col_idx + 1, depth_img.shape[1]):
    #             if depth_img[row_idx, i] > 0:
    #                 right = True
    #                 break
    #         # 왼쪽, 오른쪽 둘 다 True 일 시를 제외한 모든 픽셀 0
    #         if left == True and right == True:
    #             pass
    #         else:
    #             cutted_depths[row_idx, col_idx] = 0
    #
    # cv2.imshow('ori', depth_img)
    # cv2.imshow('complete', final_depths)
    # cv2.imshow('cutted', cutted_depths)
    # cv2.waitKey()

    ############################
    # ply_file = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation/data0349.ply')
    # mesh = vedo.load(str(ply_file))
    # # points = vedo.Points(mesh.points())
    # # print(points)
    # # points.projectOnPlane()
    # # mesh.projectOnPlane()
    # #
    # # bands = mesh.isolines()
    # # print(bands)
    # #
    # # # print(len(mesh.faces()))
    # # # print(mesh.backFaceCulling())
    # # # print(mesh.backFaceCulling())
    # #
    #
    # x_min = mesh.xbounds()[0]
    # x_max = mesh.xbounds()[1]
    # y_min = mesh.ybounds()[0]
    # y_max = mesh.ybounds()[1]
    # z_min = mesh.zbounds()[0]
    # z_max = mesh.zbounds()[1]
    #
    # x_scale = 200
    # y_scale = int(x_scale * (y_max - y_min) / (x_max - x_min))
    # depthmap = np.zeros((y_scale, x_scale), dtype=np.uint8)
    #
    # for cell_idx in range(mesh.NCells()):
    #     cell = mesh.polydata().GetCell(cell_idx)
    #     x_coord = sum(cell.GetBounds()[:2]) / len(cell.GetBounds()[:2])
    #     y_coord = sum(cell.GetBounds()[2:4]) / len(cell.GetBounds()[2:4])
    #     z_coord = sum(cell.GetBounds()[4:]) / len(cell.GetBounds()[4:])
    #     # print(f"cell real coordinate x: {x_coord}, y: {y_coord}, z: {z_coord}")
    #
    #     if (z_coord - z_min) / (z_max - z_min) < 0.4:
    #         continue
    #
    #     x = int(x_scale * (x_coord - x_min) / (x_max - x_min))
    #     y = int(y_scale * (y_coord - y_min) / (y_max - y_min))
    #     depth_value = int(255 * (z_coord - z_min) / (z_max - z_min))
    #     # print(f"relative coordinate x: {x}, y: {y}, depth value: {depth_value}")
    #
    #     depthmap[y][x] = depth_value
    #
    #     # if depthmap[y][x] > 0:

    # blur_img = cv2.GaussianBlur(depth_img, (3, 3), 0)

    kernel = np.ones((3, 3), np.float32) * 255
    dst = cv2.filter2D(depth_img, -1, kernel)

    ret, img_binary = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE,)

    print(hierarchy)

    # for cnt_idx in range(len(contours)):
    #     # if len(cnt) < 500:
    #     #     continue
    #     if hierarchy[0][cnt_idx][3] != -1:
    #         continue
    #     cv2.drawContours(depth_img, [contours[cnt_idx]], 0, (255, 0, 0), 3)

    cv2.imshow('depth map', depth_img)
    cv2.imshow('filled', final_depths)
    cv2.imshow('kernel img', dst)
    cv2.waitKey()
    # vedo.show(mesh)


if __name__ == '__main__':
    main()