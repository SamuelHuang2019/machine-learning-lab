# YOLO-v1 网络调试、训练与测试

[toc]

## 网络调试

### 合并标注文件

通过文件读写合并标注数据文件。

```python
path = '../../data/VOC2007/'
with open(path + 'label_train.txt', 'w') as fw:
    for i in range(1, 201):
        file_name = path + 'labels/' + str(i).rjust(7, '0') + '.txt'
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                fw.write(line.strip() + ' ')
            fw.write('\n')
```

此处标注数据的组成形式为`c`, `x`, `y`, `x2`, `y2`，因此需要对应修改`dataset.py`中`yoloDataset`类下的`__init__()`方法：

```python
x = float(split[1 + 5 * i])
y = float(split[2 + 5 * i])
x2 = float(split[3 + 5 * i])
y2 = float(split[4 + 5 * i])
c = split[0 + 5 * i]
```

### 格式化代码

原作者的代码中充斥大量typo、重复代码、单行代码过长、在 `docstring` 中使用单引号等不符合 `The Zen of Python` 的代码片段，且均没有进行格式化。为提升代码可读性，有必要进行相应的修复和优化。

### 补全`yoloDataset.encoder()`方法

依照原作者[@abeardear](https://github.com/abeardear)提交在[仓库 abeardear/pytorch-YOLO-v1](https://github.com/abeardear/pytorch-YOLO-v1)中的源代码，修改`dataset.py`中`yoloDataset`类下的`encoder()`方法如下：

```python
def encoder(self, boxes, labels):
    """
    implement the encoder
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 7x7x30
    """

    grid_num = 14
    target = torch.zeros((grid_num, grid_num, 30))
    cell_size = 1. / grid_num
    wh = boxes[:, 2:] - boxes[:, :2]
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample / cell_size).ceil() - 1
        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1

        xy = ij * cell_size
        delta_xy = (cxcy_sample - xy) / cell_size
        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy

    return target
```

---

### 修复图片文件无法读取的问题

此时运行网络训练，将会出现报错：

```powershell
...

File "<__array_function__ internals>", line 5, in fliplr
  File "C:\ProgramData\Anaconda3\lib\site-packages\numpy\lib\twodim_base.py", line 93, in fliplr
    raise ValueError("Input must be >= 2-d.")
ValueError: Input must be >= 2-d.
```

即，在`random_flip()`数据预处理中传入的图片数组不满足`numpy.fliplr()`的维度大于等于2的要求。

或：

```powershell
...

File "C:\Users\Guanc\Documents\GitHub\machine-learning-lab\src\YOLO-v1\dataset.py", line 121, in BGR2HSV
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.error: OpenCV(4.0.1) C:\ci\opencv-suite_1573470242804\work\modules\imgproc\src\color.cpp:181: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
```

或

```powershell
...

  File "C:/Users/Guanc/Documents/GitHub/machine-learning-lab/src/YOLO-v1/dataset.py", line 205, in randomCrop
    height, width, c = bgr.shape
AttributeError: 'NoneType' object has no attribute 'shape'
```

等等。以上错误均出现在数据预处理环节，因各预处理执行与否为随机，因此错误也将随机出现。以上两个问题均是由图片路径为空造成，为此，在调用数据预处理方法代码前设置断点：

```python
>>> if self.train:
        img, boxes = self.random_flip(img, boxes)
        img, boxes = self.randomScale(img, boxes)
        img = self.randomBlur(img)
        img = self.RandomBrightness(img)
        img = self.RandomHue(img)
        img = self.RandomSaturation(img)
        img, boxes, labels = self.randomShift(img, boxes, labels)
        img, boxes, labels = self.randomCrop(img, boxes, labels)
```

通过debugger观察到，此时，图片文件名变量`fname`值为`{str} '0'`，且`yoloDataset`对象成员变量`fnames`值为`{list: 180} ['0', ..., '0']`，读取到的图片`image`也为`None`。在对象的初始化方法中，`fnames`被初始化为

```python
self.fnames = []
...
self.fnames.append(split[0])
```

可见，原作者设计的数据集读取方式与我们所指定的有所不同，图片文件名本应出现在标注数据的第一列。重新定义`fnames`列表如下：

```python
idx = 1
for line in lines:
    split = line.strip().split()
    self.fnames.append(str(idx).rjust(7, '0') + '.jpg')
    idx += 1
    ...
```

即解决了图片文件无法读取的问题。

---

### 解决部分 `PyTorch` 语法被废弃所引起的用户警告

#### `nn.functional.sigmoid` 被废弃

运行`train.py`，终端输出大量错误警告：

```powershell
UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
  warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
```

原作者并未显式调用`nn.functional.sigmoid`，因此该警告的原因尚无从得知。

#### 以整型进行布尔索引被废弃

```powershell
UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at  ..\aten\src\ATen/native/IndexingUtils.h:25.)
```

该数据类型警告源于作者使用整型值`0`与`1`表征物体的存在与否，并基于此对预测值、实际值张量进行布尔型索引，而通过整型进行布尔索引将在随后的版本中被废弃。为此，只需将对应的张量转型为`torch.bool`类即可。

```python
coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor).bool()
noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor).bool()

noo_pred_mask = noo_pred_mask.bool()

coo_not_response_mask.zero_()
coo_not_response_mask = coo_not_response_mask.bool()
```

>值得注意的是，`PyTorch` 中并不支持类似 `NumPy` 中 `ndarray.astype()` 的张量数据类型转换，其函数返回值为转型后的张量，而原张量不变，因而必须按以下形式进行类型转换：
>
>```python
>new_tensor = tensor.bool()
>```

由于随后在计算损失函数时将会调用上述几个张量，因此，仍然需要将其转回`0`、`1`整型。

---

#### `mse_loss()` 的 `size_average` 参数将被废弃

```powershell
C:\ProgramData\Anaconda3\lib\site-packages\torch\nn\_reduction.py:44: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
```

将对应的 `size_average=False` 参数改为 `reduction='sum'` 即可。

### 修复损失函数不可用的问题

运行`train.py`，此时训练已能运行，但每次迭代中的损失函数均为 `nan`。

```powershell
Epoch [1/10], Iter [5/180] Loss: nan, average_loss: nan
Epoch [1/10], Iter [10/180] Loss: nan, average_loss: nan
Epoch [1/10], Iter [15/180] Loss: nan, average_loss: nan
Epoch [1/10], Iter [20/180] Loss: nan, average_loss: nan
...
```

在`yoloLoss.forward()`的末尾设置断点：

```python
>>> return (self.l_coord * loc_loss +
            2 * contain_loss + not_contain_loss +
            self.l_noobj * noobj_loss +
            class_loss) / N
```

通过debugger观察到，损失函数的组成部分中，`loc_loss`为`{Tensor} tensor(nan, grad_fn=<AddBackward0>)`，导致最终的损失函数为`nan`。注意到`loc_loss`由两部分构成，将其分离进行调试。

```python
loc_loss_1 = F.mse_loss(box_pred_response[:, :2],
                        box_target_response[:, :2],
                        reduction='sum')
loc_loss_2 = F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]),
                        torch.sqrt(box_target_response[:, 2:4]),
                        reduction='sum')
loc_loss = loc_loss_1 + loc_loss_2
# loc_loss = F.mse_loss(box_pred_response[:, :2],
#                       box_target_response[:, :2],
#                       reduction='sum') + \
#            F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]),
#                       torch.sqrt(box_target_response[:, 2:4]),
#                       reduction='sum')
```

观察到，`loc_loss_1`正常，而`loc_loss_2`值为`nan`，而用以计算`loc_loss_2`的张量`box_target_response`中出现了小于零的值，其作为`torch.sqrt()`的入参时自然会产生数值计算错误。虽然这部分负值出现的原因未知，但注意到其绝对值都极小，可以直接化为0处理。则代码对应修改为：

```python
F.mse_loss(torch.sqrt(box_pred_response[:, 2:4].clamp(min=0)),
                              torch.sqrt(box_target_response[:, 2:4].clamp(min=0)),
                              reduction='sum')
```

至此，训练已可正常进行。

---

## 网络训练

## 网络测试
