from pathlib import Path
import shutil


if __name__ == '__main__':
    src_dir = Path('/nfs/DataArchive/DCPR-GAN-Data/DCPR-GAN_Depth')
    trgt_dir = Path('/nfs/DataArchive/DCPR-GAN-Data/DCPR-GAN_PCD_Data_refined')
    save_dir = Path('/nfs/DataArchive/DCPR-GAN-Data/DCPR-GAN_Depth_refined')
    trgt_suff = '.ply'

    for src_situ_dir in src_dir.iterdir():
        trgt_situ_dir = trgt_dir.joinpath(src_situ_dir.name)
        save_situ_dir = save_dir.joinpath(src_situ_dir.name)
        save_situ_dir.mkdir(exist_ok=True, parents=True)
        for src_img_path in src_situ_dir.iterdir():
            img_stem = src_img_path.stem

            trgt_data_path = trgt_situ_dir.joinpath(img_stem + trgt_suff)
            if trgt_data_path.is_file():
                shutil.copy(src_img_path, save_situ_dir)