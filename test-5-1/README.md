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

结果  


```
2023-06-07 03:14:22.686056: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-06-07 03:14:22.686092: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
/opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_addons/utils/tfa_eol_msg.py:23: UserWarning: 

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 

For more information see: https://github.com/tensorflow/addons/issues/2807 

  warnings.warn(
/opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.10.0 and strictly below 2.13.0 (nightly versions are not supported). 
 The versions of TensorFlow you are currently using is 2.8.4 and is not supported. 
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
/opt/conda/envs/tf/lib/python3.8/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
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

  
    
```
Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
228818944/228813984 [==============================] - 1s 0us/step
228827136/228813984 [==============================] - 1s 0us/step
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
2023-06-07 03:15:03.929523: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/envs/tf/lib/python3.8/site-packages/cv2/../../lib64:
2023-06-07 03:15:03.929572: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-06-07 03:15:03.929603: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-a62658): /proc/driver/nvidia/version does not exist
2023-06-07 03:15:03.943671: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
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
2023-06-07 03:15:19.288496: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
2023-06-07 03:15:19.607574: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
2023-06-07 03:15:19.643181: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
2023-06-07 03:15:19.667564: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 25690112 exceeds 10% of free system memory.
2023-06-07 03:15:19.680005: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 154140672 exceeds 10% of free system memory.
103/103 [==============================] - 61s 572ms/step - loss: 0.8610 - accuracy: 0.7773
Epoch 2/5
103/103 [==============================] - 54s 519ms/step - loss: 0.6534 - accuracy: 0.8981
Epoch 3/5
103/103 [==============================] - 54s 523ms/step - loss: 0.6158 - accuracy: 0.9208
Epoch 4/5
103/103 [==============================] - 53s 513ms/step - loss: 0.6002 - accuracy: 0.9257
Epoch 5/5
103/103 [==============================] - 54s 526ms/step - loss: 0.5875 - accuracy: 0.9329
```  

### 第三步：评估模型  
```
loss, accuracy = model.evaluate(test_data)
```   

```
12/12 [==============================] - 8s 504ms/step - loss: 0.6016 - accuracy: 0.9264
```   


### 第四步，导出Tensorflow Lite模型  
```
model.export(export_dir='.')
```  

```
2023-06-07 03:21:12.052033: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
INFO:tensorflow:Assets written to: /tmp/tmp7dqbscds/assets
INFO:tensorflow:Assets written to: /tmp/tmp7dqbscds/assets
2023-06-07 03:21:16.707629: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2023-06-07 03:21:16.707815: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2023-06-07 03:21:16.768244: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
  function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 27.418ms.
  function_optimizer: function_optimizer did nothing. time = 0.018ms.

/opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn("Statistics for quantized inputs were expected, but not "
2023-06-07 03:21:18.015325: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
2023-06-07 03:21:18.015377: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
INFO:tensorflow:Label file is inside the TFLite model with metadata.
fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
INFO:tensorflow:Label file is inside the TFLite model with metadata.
INFO:tensorflow:Saving labels in /tmp/tmp0uohu4eo/labels.txt
INFO:tensorflow:Saving labels in /tmp/tmp0uohu4eo/labels.txt
INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
```  













