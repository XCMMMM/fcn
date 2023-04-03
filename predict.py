import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # 注意预测过程中不会使用辅助分类器
    aux = False  # inference time not need aux_classifier
    classes = 20
    # 这里是训练过程中保存的权重，注意改成自己训练的权重的名称
    weights_path = "./save_weights/model_29.pth"
    # 预测图片，保存在当前路径下
    img_path = "./test.jpg"
    # 标签文件对应的调色板，比如预测类别为0的目标用什么颜色表示
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # 创建模型
    model = fcn_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    # 载入模型权重
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    # 将模型权重中有关辅助分类器的信息删掉
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]
    # load weights
    # 将权重载入到模型中
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    # 读取预测图片
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    # 将图片进行预处理
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    # 增加一个batch_size的维度
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        # 传入一个像素值全为0的图片对模型进行initial
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        # 将我们要预测的图片载入到我们的设备当中，传入到我们的模型里进行预测
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        # 将预测结果当中对应于主输出上的数据提取出来，用argmax(1)找到每个像素预测的所属类别，通过squeeze方法将batch_size维度压缩掉
        prediction = output['out'].argmax(1).squeeze(0)
        # 将预测结果转到cpu设备当中，转化为numpy格式，再转成uint8形式
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 读取prediction
        mask = Image.fromarray(prediction)
        # 设置他的调色板
        mask.putpalette(pallette)
        # 保存图片
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
