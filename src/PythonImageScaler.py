import os
from PIL import Image

def convertImages(file_name_list, target_image_size, src, dest, resample_mode):
    for file_name in file_name_list:
        image = Image.open(os.path.join(src, file_name))
        resized_image = image.resize(target_image_size, resample=resample_mode)
        resized_image.save(os.path.join(dest, file_name))
        print("{} converted".format(file_name))

if __name__ == "__main__":
    train_dir = "chest_xray/train/"
    test_dir = "chest_xray/test/"
    scaled_train_dir = "scaled_chest_xray/train/"
    scaled_test_dir = "scaled_chest_xray/test/"

    train_pneu = os.listdir(os.path.join(train_dir, 'PNEUMONIA'))
    train_normal = os.listdir(os.path.join(train_dir, 'NORMAL'))
    test_pneu = os.listdir(os.path.join(test_dir, 'PNEUMONIA'))
    test_normal = os.listdir(os.path.join(test_dir, 'NORMAL'))

    WIDTH = 224
    SIZE = (WIDTH,WIDTH)
    RESAMPLE_MODE = Image.LANCZOS

    # convertImages(test_normal, SIZE, os.path.join(test_dir, 'NORMAL'), os.path.join(scaled_test_dir, 'NORMAL'), RESAMPLE_MODE)
    convertImages(test_pneu, SIZE, os.path.join(test_dir, 'PNEUMONIA'), os.path.join(scaled_test_dir, 'PNEUMONIA'), RESAMPLE_MODE)
    convertImages(train_normal, SIZE, os.path.join(train_dir, 'NORMAL'), os.path.join(scaled_train_dir, 'NORMAL'), RESAMPLE_MODE)
    convertImages(train_pneu, SIZE, os.path.join(train_dir, 'PNEUMONIA'), os.path.join(scaled_train_dir, 'PNEUMONIA'), RESAMPLE_MODE)