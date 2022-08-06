import os
import os.path as path

import matplotlib.pyplot as plt
import yaml
import torch
import numpy as np
import SimpleITK as sitk
import random
import shutil

from tqdm import tqdm
from PIL import Image
from adn.utils import read_dir, get_connected_components
from collections import defaultdict
from torchvision.utils import make_grid
import xlrd


def get_artifact_dict():
    file_path = './MAR 数据统计-train.xlsx'
    wb = xlrd.open_workbook(file_path)
    ws = wb.sheet_by_index(0)
    artifact_dict = {}
    for row in range(1, ws.nrows):
        patient_num = str(int(ws.cell(row, 0).value))
        artifact_dict[patient_num] = {}
        artifact_list = []
        no_artifact_list = []
        bed_artifact = []
        for i in range(13):
            if ws.cell(row, 3*i+2).value != '':
                start = ws.cell(row, 3*i+2).value
                end = ws.cell(row, 3*i+2+1).value
                artifact_list.extend([j for j in range(int(start), int(end) + 1)])
                pass
                # print(ws.cell(row, 3*i+2+2))
        if ws.cell(row, 41).ctype == 2:
            bed_artifact = [i for i in range(int(ws.cell(row, 41).value), int(ws.cell(row, 42).value) + 1)]
            pass
        artifact_dict[patient_num]['bed_artifact'] = bed_artifact
        artifact_dict[patient_num]['artifact_list'] = artifact_list
    return artifact_dict


def make_thumbnails(images):
    images = torch.tensor(np.array(images).astype(float))[:, np.newaxis, ...]
    images = (images - images.min()) / (images.max() - images.min())
    num_rows = int(len(images) ** 0.5)
    image = make_grid(
        images, nrow=images.shape[0] // num_rows, normalize=False)
    image = image.numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image


def split_image():
    config_file = "config/dataset.yaml"
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['brachy_ct']

    patient_dirs = read_dir(config['raw_dir'], recursive=False)

    image_size = config['image_size']
    if type(image_size) is not list:
        image_size = [image_size] * 2
    thumbnail_size = config['thumbnail_size']
    if type(thumbnail_size) is not list:
        thumbnail_size = [thumbnail_size] * 2

    artifact_dict = get_artifact_dict()
    for patient_dir in tqdm(patient_dirs):
        patient_name = path.basename(patient_dir)
        if int(patient_name) > 520:
            continue
        if patient_name not in artifact_dict.keys():
            continue
        bed_artifact = artifact_dict[patient_name]['bed_artifact']
        artifact_list = artifact_dict[patient_name]['artifact_list']
        volume_files = read_dir(patient_dir, predicate=lambda x: x.startswith("CT"), recursive=True)
        # volume_files = [patient_dir]
        for volume_file in volume_files:
            volume_obj = sitk.ReadImage(volume_file)

            volume = sitk.GetArrayFromImage(volume_obj)
            # plt.imshow(volume[90])
            # plt.show()
            # volume_name = path.basename(volume_file).split(".")[0]
            volume_name = "0"
            thumbnails = defaultdict(list)
            index = 0
            for image in tqdm(volume, desc="Preparing {}_{}".format(patient_name, volume_name)):
                image_type = "no_artifact"

                # Check if the image has metal artifacts
                # if image.max() > config["max_hu"][1]:
                #     points = np.array(np.where(image > config["max_hu"][1])).T
                #     points = set(tuple(p) for p in points)
                #     components = get_connected_components(points)
                #     max_area = max(len(c) for c in components)
                #
                #     if max_area > config["connected_area"]:
                #         image_type = "artifact"
                #     else:
                #         continue
                # elif image.max() > config["max_hu"][0]:
                #     continue
                if index + 1 in bed_artifact:
                    continue
                elif index + 1 in artifact_list:
                    image_type = 'artifact'

                output_dir = path.join(config["dataset_dir"], image_type,
                                       "{}_{}".format(patient_name, volume_name))
                if not path.isdir(output_dir): os.makedirs(output_dir)

                image = Image.fromarray(image)
                image = np.array(image)
                # thumbnail = cv2.resize(image, dsize=thumbnail_size, interpolation=cv2.INTER_CUBIC)
                thumbnail = (image - image.min()) / (image.max() - image.min())
                # thumbnail = (thumbnail - thumbnail.min()) / (thumbnail.max() - thumbnail.min())
                thumbnail = (thumbnail * 255).astype(np.uint8)
                thumbnails[image_type].append(thumbnail)

                image_name = "{}_{}_{:03d}".format(patient_name, volume_name, index)
                image_file = path.join(output_dir, image_name + ".npy")
                thumbnail_file = path.join(output_dir, image_name + ".png")

                np.save(image_file, image)
                Image.fromarray(thumbnail).save(thumbnail_file)
                index += 1

            # Create an overview of images from this patient
            for k, ts in thumbnails.items():
                output_dir = path.join(
                    config["dataset_dir"], k, "{}_{}".format(patient_name, volume_name))
                if len(ts) > 0:
                    thumbnails_file = path.join(config["dataset_dir"], k, "{}_{}.png".format(patient_name, volume_name))
                    Image.fromarray(make_thumbnails(ts)).save(thumbnails_file)
                else:
                    os.removedirs(output_dir)

    # Create train and test split
    artifact_dir = path.join(config["dataset_dir"], "artifact")
    patient_dirs = read_dir(artifact_dir, "dir")
    random.shuffle(patient_dirs)

    test_patients = []
    test_cnt = 0
    index = 0
    while index < len(patient_dirs) and test_cnt < config["num_tests"]:
        num_images = len(read_dir(patient_dirs[index], "file"))
        if num_images < 100:
            test_patients.append(path.basename(patient_dirs[index]))
            test_cnt += num_images
        index += 1

    no_artifact_dir = path.join(config["dataset_dir"], "no_artifact")
    items = read_dir(artifact_dir) + read_dir(no_artifact_dir)

    test_dir = path.join(config["dataset_dir"], "test")
    train_dir = path.join(config["dataset_dir"], "train")
    if not path.isdir(test_dir): os.makedirs(test_dir)
    if not path.isdir(train_dir): os.makedirs(train_dir)

    for item in items:
        item_type, item_name = item.split(path.sep)[-2:]
        patient_name = path.splitext(item_name)[0]

        if patient_name in test_patients:
            shutil.move(item, path.join(test_dir, item_type, item_name))
        else:
            shutil.move(item, path.join(train_dir, item_type, item_name))

    shutil.rmtree(artifact_dir)
    shutil.rmtree(no_artifact_dir)
    lst = list()
    dct = dict()
    lst.sort(key='key')

if __name__ == "__main__":
    split_image()
