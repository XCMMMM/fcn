import os

import torch.utils.data as data
from PIL import Image


class VOCSegmentation(data.Dataset):
    # voc_root--指向VOCdevkit文件夹的路径
    # year--传入voc数据集的年份，该类只支持2007和2012年
    # transforms--对数据集图片进行预处理的转换
    # txt_name--是train.txt或者val.txt
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 将路径连接起来，得到最终的路径
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        # 数据集图片的存储路径
        image_dir = os.path.join(root, 'JPEGImages')
        # 数据集图片标签的存储路径（语义分割）
        mask_dir = os.path.join(root, 'SegmentationClass')
        # train.txt（训练数据集是哪些图片）或者val.txt的存储路径
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            # 读取图片名称，注意这里读取的图片名称都没有后缀
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        # 数据集每一张图片的路径
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # 数据集每一张图片标签的路径
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    # 传入索引，就可以得到对应索引的图片和标签
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # 打开图片，将图片按转化为RGB三通道格式
        img = Image.open(self.images[index]).convert('RGB')
        # 对应图片的标签，每个像素点所属类别的标签（因为不同类别的调色板不同，看palette.json文件）
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            # 对图片和标签进行预处理
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    # 用于在DataLoader函数中设置数据集打包的过程
    def collate_fn(batch):
        # images中是batch个图片文件，targets中是batch个标签文件
        images, targets = list(zip(*batch))
        # 将图片文件通过cat_list方法进行打包
        batched_imgs = cat_list(images, fill_value=0)
        # 将标签文件通过cat_list方法进行打包
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    # 训练过程中图片大小都被裁剪到480x480
    # 验证过程中只进行了缩放，没有进行裁剪，可能每张图片的大小都不一样
    # 为了将图片打包成一个tensor，所以必须用一个能装下所有图片的大的tensor
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # 在max_size前再加上batch维度
    # 例如训练时 batch_shape=(4,3,480,480)
    batch_shape = (len(images),) + max_size
    # 利用tensor的new方法构建一个维度为batch_shape的新的tensor，用fill_value填充
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # img为传入进来images中的每一张图片，pad_img为batched_imgs在batch维度的切片
    for img, pad_img in zip(images, batched_imgs):
        # 将img图片copy到pad_img中
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    # 如果是训练，最后返回的batched_imgs维度为(4,3,480,480)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
