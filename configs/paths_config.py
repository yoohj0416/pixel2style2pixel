dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
	# 'toothA_train': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel/trainA',
	# 'toothB_train': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel/trainB',
	# 'toothB_train': '/home/malab3/dataArchive/DCPR-GAN-Data/pix2pix/trainB',
	'toothA_train': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/train/3-Preparation',
	'toothB_train': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/train/2-Object',
	'tooth_prepare_train': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/train/3-Preparation',
	'tooth_opposing_train': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/train/1-Opposing_teeth',
	'tooth_object_train': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/train/2-Object',
	# 'toothA_test': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel/testA',
	# 'toothB_test': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel/testB',
	# 'toothB_test': '/home/malab3/dataArchive/DCPR-GAN-Data/pix2pix/testB',
	'toothA_test': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/test/3-Preparation',
	'toothB_test': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/test/2-Object',
	'tooth_prepare_test': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/test/3-Preparation',
	'tooth_opposing_test': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/test/1-Opposing_teeth',
	'tooth_object_test': '/nfs/DataArchive/DCPR-GAN-Data/pixel2style2pixel_inpainting/test/2-Object',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}