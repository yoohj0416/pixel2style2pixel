import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split


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


def preprocess_and_save(dataset_list, save_dir, img_size):
    for x_train_img_path in dataset_list:
        img_name = x_train_img_path.name
        img = cv2.imread(str(x_train_img_path))
        pre_img = padding(img, img_size)

        save_path = save_dir.joinpath(img_name)
        cv2.imwrite(str(save_path), pre_img)


def main():
    pre_path = Path('/home/malab4/dataArchive/DCPR-GAN-Data/3-Preparation_depth')
    obj_path = Path('/home/malab4/dataArchive/DCPR-GAN-Data/2-Object_depth')
    # save_dir = pre_path.parent.joinpath('pix2pix')
    save_dir = pre_path.parent.joinpath('pixel2style2pixel')
    save_dir.mkdir(exist_ok=True)

    x = sorted(pre_path.iterdir())
    y = sorted(obj_path.iterdir())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    img_size = 256

    save_trainA_dir = save_dir.joinpath('trainA')
    save_trainA_dir.mkdir(exist_ok=True)
    preprocess_and_save(x_train, save_trainA_dir, img_size)

    save_trainB_dir = save_dir.joinpath('trainB')
    save_trainB_dir.mkdir(exist_ok=True)
    preprocess_and_save(y_train, save_trainB_dir, img_size)

    save_testA_dir = save_dir.joinpath('testA')
    save_testA_dir.mkdir(exist_ok=True)
    preprocess_and_save(x_test, save_testA_dir, img_size)

    save_testB_dir = save_dir.joinpath('testB')
    save_testB_dir.mkdir(exist_ok=True)
    preprocess_and_save(y_test, save_testB_dir, img_size)


if __name__ == '__main__':
    main()