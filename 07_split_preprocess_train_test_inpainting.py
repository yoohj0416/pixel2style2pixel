import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


def padding(img, set_size):
    h, w, c = img.shape

    long_value = max(h, w)
    tb = int((h - long_value) / 2)
    lr = int((w - long_value) / 2)

    if tb < 0:
        tb = tb * -1
    if lr < 0:
        lr = lr * -1

    pad_img = cv2.copyMakeBorder(img, tb, tb, lr, lr, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized_img = cv2.resize(pad_img, dsize=(set_size, set_size), interpolation=cv2.INTER_AREA)

    return resized_img


def preprocess_and_save(src_img_path, save_dir, img_size):
    img_name = src_img_path.name
    img = cv2.imread(str(src_img_path))
    pre_img = padding(img, img_size)

    save_path = save_dir.joinpath(img_name)
    cv2.imwrite(str(save_path), pre_img)


def main():
    src_dir = Path('/nfs/DataArchive/DCPR-GAN-Data/DCPR-GAN_Depth_refined')
    save_dir = src_dir.parent.joinpath('pixel2style2pixel_inpainting')
    save_dir.mkdir(exist_ok=True)

    img_size = 256

    train_img_dir = src_dir.parent.joinpath('pixel2style2pixel_encode', 'trainA')
    train_img_list = [img_path.stem for img_path in train_img_dir.iterdir()]
    test_img_dir = src_dir.parent.joinpath('pixel2style2pixel_encode', 'testA')
    test_img_list = [img_path.stem for img_path in test_img_dir.iterdir()]

    save_train_dir = save_dir.joinpath('train')
    for src_situ_dir in src_dir.iterdir():
        save_situ_dir = save_train_dir.joinpath(src_situ_dir.name)
        save_situ_dir.mkdir(exist_ok=True, parents=True)
        for src_img_path in src_situ_dir.iterdir():
            if src_img_path.stem in train_img_list:
                preprocess_and_save(src_img_path, save_situ_dir, img_size)

    save_test_dir = save_dir.joinpath('test')
    for src_situ_dir in src_dir.iterdir():
        save_situ_dir = save_test_dir.joinpath(src_situ_dir.name)
        save_situ_dir.mkdir(exist_ok=True, parents=True)
        for src_img_path in src_situ_dir.iterdir():
            if src_img_path.stem in test_img_list:
                preprocess_and_save(src_img_path, save_situ_dir, img_size)


if __name__ == '__main__':
    main()