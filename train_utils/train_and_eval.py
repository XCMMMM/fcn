import torch
from torch import nn
import train_utils.distributed_utils as utils


def criterion(inputs, target):
    losses = {}
    # inputs就是模型预测的结果
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)
    # len(losses) == 1--表明没有使用辅助分类器，就将我们主输出上的loss返回
    if len(losses) == 1:
        return losses['out']
    # 使用了辅助分类器，返回 主输出上的loss+0.5*辅助分类器上的loss
    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    # 创建一个混淆矩阵（混淆矩阵也称误差矩阵，是表示精度评价的一种标准格式，用n行n列的矩阵形式来表示。）
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        # 遍历我们的验证集
        for image, target in metric_logger.log_every(data_loader, 100, header):
            # 将图片和标签指定到对应的设备当中
            image, target = image.to(device), target.to(device)
            # 将数据传入到模型中进行预测
            output = model(image)
            # 注意这里的输出只采用主分支上的输出
            output = output['out']
            # 维度顺序分别为：batch_size，channel，高度，宽度
            # 这里output.argmax(1)--对应的是找到channel维度的最大值（每个channel维度有21个值，找到预测值最大的那个）
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # criterion--计算模型输出和target之间的损失
            loss = criterion(output, target)

        # 清空优化器的历史梯度
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 将误差进行反向传播
            loss.backward()
            # 更新我们的参数
            optimizer.step()

        # 更新学习率，注意这里是每迭代一个step就会更新一下学习率（而不是每迭代一个epoch）
        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    # 返回训练后的平均损失和学习率
    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
