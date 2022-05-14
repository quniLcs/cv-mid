# 计算机视觉期中作业

## ResNet

GitHub repo
链接：[https://github.com/quniLcs/cv-mid](https://github.com/quniLcs/cv-mid)

网盘链接：[百度网盘-计算机视觉期中作业](https://pan.baidu.com/s/1aG9DZx2wrY5wyNLGOBk8LQ?pwd=exr2)

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

### 数据集

```
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

### 预训练模型

使用`Imagenet`上的预训练模型`vgg16`的卷积层参数。

文件见百度网盘。

### 训练模型

运行`model_train.ipynb`文件。

每个epoch的模型都记录在`checkpoints`中。

### 测试模型

运行`model_test.ipynb`文件。

返回包括类别标签，得分，bounding box的预测图片。

## YOLOv3

### 数据集

```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

通过`voc_label.py`，合并VOC2007，VOC2012的训练集与验证集。

### 预训练模型

使用`Imagenet`上的预训练模型`darknet53`的卷积层参数。

```
wget https://pjreddie.com/media/files/darknet53.conv.74
```
### 训练模型

```
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 | -gpu 0 tee train_yolov3.log
```

训练并记录训练日志。

### 测试模型

单张图片测试可以执行

```
./darknet detector test cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc_23500.weights <path/to/image>
```

对模型性能mIoU, mAP的验证可以调用`recall`、`validation`命令。
