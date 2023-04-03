import os
import time
import datetime

import torch

from src import fcn_resnet50
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import VOCSegmentation
import transforms as T

# 训练过程中采用的图像预处理方法
class SegmentationPresetTrain:
    # hflip_prob--图像水平翻转的概率
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        # 在min_size和max_size之间随机选取一个数值size（可能是min_size和max_size之间的任意一个数值），然后将图像的最小边长缩放到size大小
        trans = [T.RandomResize(min_size, max_size)]
        # 如果设置的随机水平翻转的概率大于0，就会将图像和对应的target进行随机的翻转
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            # 将图片进行随机的裁剪
            T.RandomCrop(crop_size),
            # 将图片转化为tensor格式，转化过程中也会将图片像素数值从0-255缩放到0-1之间
            # 也会将target转化为tensor格式，不过不会对它进行缩放
            T.ToTensor(),
            # 标准化处理
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

# 预测过程中采用的图像预处理方法
class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            # 不同于训练时，此时min_size和max_size都设置成base_size，将图像的最小边长缩放到base_size大小
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 520
    crop_size = 480
    # 训练时做了先缩放（将图像的最小边长缩放到520），再裁剪到480x480（我理解是不是做了图像增广）
    # 预测时直接将图像的最小边长缩放到520，没有进行裁剪
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)

# 传入两个参数，aux--对应于是否使用辅助分类器，num_classes--分类类别
def create_model(aux, num_classes, pretrain=True):
    # 创建模型，该函数在src文件夹下的fcn_model.py文件中
    model = fcn_resnet50(aux=aux, num_classes=num_classes)

    if pretrain:
        # 载入权重文件
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]
        # 将权重载入到模型中
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            # 打印哪些权重还没有载入
            print("missing_keys: ", missing_keys)
            # 打印哪些权重还没有用到
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    # 判断gpu设配是否可用
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    # 因为刚开始的num_classes没有加上背景类别，类别个数为20
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 调用我们在my_dataset.py下编写的数据集读取部分，读取我们的训练数据和测试数据
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")
    
    # os.cpu_count()--gpu的核数
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    # 调用上面的create_model函数，实例化模式
    model = create_model(aux=args.aux, num_classes=num_classes)
    model.to(device)

    # p.requires_grad为真，代表着权重没有被冻结，就将权重取出来
    # model.backbone--对应于fcn结构图中的Resnet50 Backbone
    # model.classifier--对应于fcn结构图中的FCN Head
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        # 如果使用辅助分类器，就将model.aux_classifier下没有被冻结的权重取出来一起训练
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        # 将辅助分类器的训练参数params添加到之前的列表params_to_optimize中
        # 注意辅助分类器的学习率是初始学习率的10倍
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # 设置优化器采用SGD，传入需要训练的权重参数params_to_optimize，学习率lr，momentum以及weight_decay
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    # warmup=True--可以理解为热身训练，会从一个非常小的学习率开始训练，慢慢增加到我们所指定的初始学习率，然后再慢慢的下降
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # 是否传入resume参数，如果传入就会载入最近一次保存的模型权重，然后读取模型权重，优化器的相关数据以及学习率更新策略的数据
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train_one_epoch--数据训练一轮的过程
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # evaluate--验证过程
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        # 保存训练相关的信息
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


# 传入的参数
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    #使用的是V0C2012数据集，这里传入数据集路径，默认数据集存放在当前fcn路径下的data文件夹中
    parser.add_argument("--data-path", default="./data/", help="VOCdevkit root")
    # 类别数，这里不包括背景维，类别数为20
    parser.add_argument("--num-classes", default=20, type=int)
    # aux--是否使用辅助分类器，这里默认使用辅助分类器
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    # 训练网络到一半中断了，可以将default设置指向为中断时的网络权重，就会接着上一次的向后训练，就不需要重新开始训练了
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # start-epoch--从第几个epoch开始训练，默认从0开始
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
