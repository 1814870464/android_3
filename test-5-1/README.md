# 使用TensorFlow Lite Model Maker生成图像分类模型

### 1. 预备工作  

首先安装程序运行必备的一些库。
```python
!pip install tflite-model-maker
```  

接下来，导入相关的库。  

```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```  

## 模型训练  

### 获取数据  
本实验先从较小的数据集开始训练，当然越多的数据，模型精度更高。  
```
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
```
## 运行示例  

一共需4步完成。  

### 第一步：加载数据集，并将数据集分为训练数据和测试数据。  
```
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```  
结果  
```
INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
```  
### 第二步：训练Tensorflow模型  
```
model = image_classifier.create(train_data)
```  
结果  
```
INFO:tensorflow:Retraining the models...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
 rasLayerV1V2)                                                   
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 5)                 6405      
                                                                 
=================================================================
Total params: 3,419,429
Trainable params: 6,405
Non-trainable params: 3,413,024
_________________________________________________________________
None
Epoch 1/5


d:\anaconda3\lib\site-packages\keras\optimizer_v2\gradient_descent.py:102: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
  super(SGD, self).__init__(name, **kwargs)


103/103 [==============================] - 76s 719ms/step - loss: 0.8647 - accuracy: 0.7782
Epoch 2/5
103/103 [==============================] - 97s 943ms/step - loss: 0.6525 - accuracy: 0.8935
Epoch 3/5
103/103 [==============================] - 92s 896ms/step - loss: 0.6223 - accuracy: 0.9099
Epoch 4/5
103/103 [==============================] - 95s 921ms/step - loss: 0.6021 - accuracy: 0.9226
Epoch 5/5
103/103 [==============================] - 100s 970ms/step - loss: 0.5903 - accuracy: 0.9336
```  

### 第三步：评估模型  
```
loss, accuracy = model.evaluate(test_data)
loss, accuracy = model.evaluate(test_data)
```   

```
12/12 [==============================] - 12s 749ms/step - loss: 0.6107 - accuracy: 0.9155
```   


### 第四步，导出Tensorflow Lite模型  
```
model.export(export_dir='.')
```  

```
INFO:tensorflow:Assets written to: C:\Users\ll\AppData\Local\Temp\tmpqryprhqv\assets


INFO:tensorflow:Assets written to: C:\Users\ll\AppData\Local\Temp\tmpqryprhqv\assets
d:\anaconda3\lib\site-packages\tensorflow\lite\python\convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn("Statistics for quantized inputs were expected, but not "


INFO:tensorflow:Label file is inside the TFLite model with metadata.


INFO:tensorflow:Label file is inside the TFLite model with metadata.


INFO:tensorflow:Saving labels in C:\Users\ll\AppData\Local\Temp\tmpjyprgcyp\labels.txt


INFO:tensorflow:Saving labels in C:\Users\ll\AppData\Local\Temp\tmpjyprgcyp\labels.txt


INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite


INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
```  













