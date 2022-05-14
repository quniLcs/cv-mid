# 计算机视觉期中作业

## ResNet

GitHub repo
链接：[https://github.com/quniLcs/cv-mid](https://github.com/quniLcs/cv-mid)

网盘链接：[https://pan.baidu.com/s/1aG9DZx2wrY5wyNLGOBk8LQ?pwd=exr2](https://pan.baidu.com/s/1aG9DZx2wrY5wyNLGOBk8LQ?pwd=exr2)

### 使用模块说明

`argparse`：用于从命令行设置超参数；

`numpy`：用于数学计算；

`torch`：用于搭建并训练网络；

`torchvision`：用于加载数据集；

`matplotlib.pyplot`：用于可视化。

### 代码文件说明

`load.py`：定义一个函数`load`，输入五个参数，第一个参数表示数据集文件路径，第二个参数表示是否训练集，第三个参数表示批量大小，第四个参数表示是否打乱顺序，第五个参数表示子进程数量；直接运行该文件时，调用该函数，保存训练集的前三张样本。

`cutout.py`：定义一个函数`cutout`，输入三个参数，第一个参数表示一批图像，第二个参数表示正方形边长，第三个参数表示硬件；直接运行该文件时，调用`load`加载数据集，并调用该函数，保存处理后的前三张样本。

`mixup.py`：定义一个函数`mixup`，输入三个参数，第一个参数表示一批图像，第二个参数表示一批标签，第三个参数表示Beta分布的参数；直接运行该文件时，调用`load`加载数据集，并调用该函数，保存处理后的前三张样本。

`cutmix.py`：定义一个函数`cutmix`，输入三个参数，含义与函数`mixup`相同；直接运行该文件时，调用`load`加载数据集，并调用该函数，保存处理后的前三张样本。

`model.py`：定义模型`ResNet`。

`util.py`：定义四个函数，`optimize`用于使用指定的数据增强方法训练一个回合的模型，`evaluate`用于在训练集或测试集上评估模型，`save_status`用于保存模型和优化器，`load_status`用于加载模型。

`main.py`：调用`load`加载训练集和测试集，实例化`ResNet`，使用学习率阶梯下降且带有动量的随机梯度下降优化器、交叉熵损失函数、调用`optimize`和`evaluate`训练并测试模型，最后调用`save_status`保存模型和优化器。

### 训练和测试示例代码

在命令行使用不同的数据增强方法训练模型：

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python main.py --mode baseline
python main.py --mode cutout
python main.py --mode mixup
python main.py --mode cutmix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

其它命令行超参数：

`batch_size`：表示批量大小，默认值`128`；

`num_epoch`：表示回合数，默认值`80`；

`lr`：表示初始学习率，默认值`0.1`；

`milestones`：表示学习率下降回合列表，默认值`[20, 40, 60]`；

`gamma`：表示学习率下降参数，默认值`0.2`；

`momentum`：表示动量，默认值`0.9`；

`lambd`：表示正则化参数，默认值`5e-4`。

## Faster R-CNN


