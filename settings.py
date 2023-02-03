#Data Processing
STAGE1_CATEGORIES = ['healthy', 'damaged']
STAGE2_CATEGORIES = ['anthracnose', 'crown_rot', 'fruit_sooty_molds']

stage1_fileoutput = './stage1/'
stage2_fileoutput = './stage2/'

picture_width = 640
picture_height = 480

#Training Stage 1 Model Settings
stage1_no_dense_layer = [1, 2, 3, 4]
stage1_no_conv_layer = [1, 2, 3, 4, 5]
stage1_dense_layer_size = [32, 64, 128]
stage1_conv_layer_size = [32, 64, 128, 256]

#Training Stage 2 Model Settings
stage2_no_dense_layer = [1, 2, 3, 4]
stage2_no_conv_layer = [1, 2, 3, 4, 5]
stage2_dense_layer_size = [32, 64, 128]
stage2_conv_layer_size = [32, 64, 128, 256]
