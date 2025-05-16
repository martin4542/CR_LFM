import os
import random
import numpy as np
import tifffile as tiff

import torch.utils.data as data
import torchvision.transforms as transforms


IMAGE_SIZE = 256


class SEN12Dataset(data.Dataset):
    def __init__(self, root, mode="train", random_flip=False, image_size=IMAGE_SIZE):
        self.mode = mode
        self.root_dir = os.path.join(root, mode)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.random_flip = random_flip

        self.src_img_path = os.path.join(self.root_dir, "s2_cloudy")
        self.target_img_path = os.path.join(self.root_dir, "s2")
        self.ridar_iamage_path = os.path.join(self.root_dir, "s1")

        self.__load_dataset()

    def __load_dataset(self):
        src_imgs = sorted(os.listdir(self.src_img_path))
        target_imgs = sorted(os.listdir(self.target_img_path))
        ridar_imgs = sorted(os.listdir(self.ridar_iamage_path))

        assert len(src_imgs) == len(target_imgs) == len(ridar_imgs), "The number of images in the source, target, and ridar directories must be the same."

        self.src_imgs = [os.path.join(self.src_img_path, img) for img in src_imgs]
        self.target_imgs = [os.path.join(self.target_img_path, img) for img in target_imgs]
        self.ridar_imgs = [os.path.join(self.ridar_iamage_path, img) for img in ridar_imgs]

    def __getitem__(self, index):
        images = [
            tiff.imread(self.src_imgs[index]),
            tiff.imread(self.target_imgs[index]),
            tiff.imread(self.ridar_imgs[index])
        ]
        images =[image[:,:,:3] for image in images]

        if self.random_flip:
            random_prob = random.random()
            if random_prob <= 0.25:
                flip_parameter = random.randint(0, 2)
                images = [np.flip(image, flip_parameter) for image in images]
            elif 0.25 < random_prob <= 0.5:
                rotate_parameter = random.randint(1, 3)
                images = [np.rot90(image, rotate_parameter) for image in images]
        
        images = [self.transform(image) for image in images]

        return images[0], images[1], images[2]

    def __len__(self):
        return len(self.src_imgs)




if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/workspace/generative_model/data", help="Root directory of the dataset")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"], help="Mode of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloader")
    args = parser.parse_args()

    dataset = SEN12Dataset(root=args.root, mode=args.mode, random_flip=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for i, (src_img, target_img, ridar_img) in enumerate(dataloader):
        print(src_img.shape)
        print(target_img.shape)
        print(ridar_img.shape)
        break