from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from depth_map_utils import fill_in_multiscale


def main():
    ply_file = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation/data0349.ply')
    # ply_file = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation/data0358.ply')
    pcd = o3d.io.read_point_cloud(str(ply_file))
    pts = np.asarray(pcd.points)

    print(f'num. of points: {len(pts)}')
    print(f'shape of points: {pts.shape}')
    xscale = 200
    xmax = np.max(pts[:, 0])
    xmin = np.min(pts[:, 0])
    ymax = np.max(pts[:, 1])
    ymin = np.min(pts[:, 1])
    zmax = np.max(pts[:, 2])
    zmin = np.min(pts[:, 2])
    yscale = int(xscale * (ymax - ymin) / (xmax - xmin))
    print(f"x length: {xmax - xmin}")
    print(f'xscale {xscale}, yscale {yscale}')
    depthmap = np.zeros((yscale + 2, xscale + 2), dtype=np.uint8)

    for pt in pts:
        x, y, z = pt

        if (z - zmin) / (zmax - zmin) < 0.5:
            continue

        x = int(xscale * (x - xmin) / (xmax - xmin))
        y = int(yscale * (y - ymin) / (ymax - ymin))
        depthmap[y][x] = int(255 * (z - zmin) / (zmax - zmin))

    # thrsh_depthmap = depthmap.copy()
    # thrsh_depthmap[(thrsh_depthmap / 255) < 0.5] = 0

    projected_depths = np.float32(depthmap / 255.0)
    final_depths, process_dict = fill_in_multiscale(projected_depths,
                                                    max_depth=255,
                                                    extrapolate=False,
                                                    blur_type='bilateral')

    kernel = np.ones((9, 9), np.float32) * 255
    dst = cv2.filter2D(depthmap, -1, kernel)

    ret, img_binary = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE, )

    outer_contours = [contours[cnt_idx] for cnt_idx in range(len(contours)) if hierarchy[0][cnt_idx][3] == -1]
    # for cnt in outer_contours:
    #     cv2.drawContours(depthmap, [cnt], 0, (255, 0, 0), 3)

    sliced_depthmap = np.zeros_like(final_depths, dtype=np.float32)
    for row_idx in range(depthmap.shape[0]):
        for col_idx in range(depthmap.shape[1]):
            for cnt in outer_contours:
                if cv2.pointPolygonTest(cnt, (col_idx, row_idx), False) > 0:
                    sliced_depthmap[row_idx, col_idx] = final_depths[row_idx, col_idx]

    # resized_depthmap = cv2.resize(depthmap, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # resized_depthmapx4 = cv2.resize(depthmap, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('depthmap', depthmap)
    # cv2.imshow('resized', resized_depthmap)
    # cv2.imshow('resizedx4', resized_depthmapx4)
    # cv2.imshow('thrsh depthmap', thrsh_depthmap)
    # cv2.imshow('kernel', dst)
    cv2.imshow('filled', final_depths)
    cv2.imshow('sliced', sliced_depthmap)
    cv2.waitKey()


if __name__ == '__main__':
    main()