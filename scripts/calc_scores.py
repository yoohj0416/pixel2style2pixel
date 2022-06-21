import argparse
import cv2
from pathlib import Path
import math
import numpy as np
from IQA_pytorch import FSIM, utils
from image_similarity_measures.quality_metrics import fsim

from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Output image directory")
    parser.add_argument('--gt_dir', type=str, help="Ground truth image directory")
    parser.add_argument('--in_dir', type=str, help="Input image directory")
    parser.add_argument('--save_dir', type=str, help="Directory path to save result of comparison")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    out_dir = Path(args.out_dir)
    gt_dir = Path(args.gt_dir)
    in_dir = Path(args.in_dir)

    img_size = 256
    MAX_PIX = 255

    rmse_list = []
    psnr_list = []
    ssim_list = []
    fsim_list = []

    for out_img_path in out_dir.iterdir():
        out_img_name = out_img_path.name
        gt_img_path = gt_dir.joinpath(out_img_name)
        assert gt_img_path.is_file(), f"There is incorrect file: {gt_img_path}"

        out_img = cv2.imread(str(out_img_path), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(str(gt_img_path), cv2.IMREAD_GRAYSCALE)
        out_img = cv2.resize(out_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gt_img = cv2.resize(gt_img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # Normalize image 0 to 1
        out_img_norm = cv2.normalize(out_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        gt_img_norm = cv2.normalize(gt_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # calculate RMSE
        mse = mean_squared_error(gt_img_norm, out_img_norm)
        rmse_list.append(mse ** 0.5)

        # calculate PSNR
        if mse == 0:
            psnr_list.append(100.0)
        else:
            # psnr_list.append(20 * math.log10(MAX_PIX / math.sqrt(mse)))
            psnr_list.append(20 * math.log10(1.0 / math.sqrt(mse)))

        # calculate SSIM
        ssim_score, _ = ssim(gt_img_norm, out_img_norm, full=True)
        ssim_list.append(ssim_score)

        # calculate FSIM
        gt_img_norm_expand = np.expand_dims(gt_img_norm, axis=2)
        out_img_norm_expand = np.expand_dims(out_img_norm, axis=2)
        fsim_list.append(fsim(gt_img_norm_expand, out_img_norm_expand))
        # fsim = FSIM(channels=1)
        # out_tensor = utils.prepare_image(out_img_norm)
        # gt_tensor = utils.prepare_image(gt_img_norm)
        #
        # fsim_list.append(fsim(out_tensor, gt_tensor, as_loss=False))

        # save result image of comparison
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

            in_img_path = in_dir.joinpath(out_img_name)
            in_img = cv2.imread(str(in_img_path), cv2.IMREAD_GRAYSCALE)
            in_img = cv2.resize(in_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            compare_img = cv2.hconcat([in_img, gt_img, out_img])

            under_90_dir = save_dir.joinpath("90-")
            under_90_dir.mkdir(exist_ok=True)
            between_95_90_dir = save_dir.joinpath("90-95")
            between_95_90_dir.mkdir(exist_ok=True)
            upper_95_dir = save_dir.joinpath("-95")
            upper_95_dir.mkdir(exist_ok=True)

            compare_img_name = out_img_path.stem + f"_ssim{ssim_score:.3f}" + '.png'
            if ssim_score < 0.9:
                compare_img_path = under_90_dir.joinpath(compare_img_name)
            elif 0.9 <= ssim_score < 0.95:
                compare_img_path = between_95_90_dir.joinpath(compare_img_name)
            else:
                compare_img_path = upper_95_dir.joinpath(compare_img_name)
            # cv2.imshow('compare_img', compare_img)
            # cv2.waitKey()
            cv2.imwrite(str(compare_img_path), compare_img)

    print(f"RMSE score: {sum(rmse_list) / len(rmse_list):.4f}")
    print(f"PSNR score: {sum(psnr_list) / len(psnr_list):.4f}")
    print(f"SSIM score: {sum(ssim_list) / len(ssim_list):.4f}")
    print(f"FSIM score: {sum(fsim_list) / len(fsim_list):.4f}")


if __name__ == '__main__':
    main()