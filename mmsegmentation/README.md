# 训练过程
## 准备阶段
1. config文件![](./pspnet-Watermelon_20230618.py)
2. dataset文件![](./WatermenlonDataset.py)

## 开始训练
`python train.py`

## 可视化训练日志
`python TrainingLog.py`

# 训练结果
训练日志的csv文件在Result_in_Training中
## 训练集结果：
![](./Result_in_Training/训练集损失函数.png)
![](./Result_in_Training/训练集准确率.png)
## 测试集指标：
![](./Result_in_Training/val_results.jpg)
![](./Result_in_Training/测试集分类评估指标.png)

# 个人数据上推理测试
## 图像上的分割
输入图像![](./TestDataSet/watermelon.jpg)
推理结果![](./TestDataResults/watermelon.jpg)
## 视频上的分割
输入视频![](./TestDataSet/watermelon.mp4)
推理结果![](./TestDataResults/watermelon_out.mp4)
