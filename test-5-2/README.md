# TensorFlow训练石头剪刀布数据集  
本文将演示石头剪刀布图片库的神经网络训练过程。石头剪刀布数据集包含了不同的手势图片，来自不同的种族、年龄和性别。  

### 首先下载石头剪刀布的训练集和测试集：  
```
!wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps.zip
  
!wget --no-check-certificate https://storage.googleapis.com/learning-datasets/rps-test-set.zip
```  

```
--2023-06-07 03:46:38--  https://storage.googleapis.com/learning-datasets/rps.zip
Resolving storage.googleapis.com (storage.googleapis.com)... 142.250.31.128, 142.251.111.128, 142.251.16.128, ...
Connecting to storage.googleapis.com (storage.googleapis.com)|142.250.31.128|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 200682221 (191M) [application/zip]
Saving to: ‘rps.zip’

rps.zip             100%[===================>] 191.38M  90.8MB/s    in 2.1s    

2023-06-07 03:46:40 (90.8 MB/s) - ‘rps.zip’ saved [200682221/200682221]

--2023-06-07 03:46:41--  https://storage.googleapis.com/learning-datasets/rps-test-set.zip
Resolving storage.googleapis.com (storage.googleapis.com)... 142.251.163.128, 142.250.31.128, 172.253.62.128, ...
Connecting to storage.googleapis.com (storage.googleapis.com)|142.251.163.128|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 29516758 (28M) [application/zip]
Saving to: ‘rps-test-set.zip’

rps-test-set.zip    100%[===================>]  28.15M   102MB/s    in 0.3s    

2023-06-07 03:46:41 (102 MB/s) - ‘rps-test-set.zip’ saved [29516758/29516758]
```  


### 然后解压下载的数据集。

```
import os
import zipfile

local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

local_zip = 'rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()
```  

### 检测数据集的解压结果，打印相关信息。  
```
rock_dir = os.path.join('rps/rock')
paper_dir = os.path.join('rps/paper')
scissors_dir = os.path.join('rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10]
```  

```
total training rock images: 840
total training paper images: 840
total training scissors images: 840
['rock07-k03-058.png', 'rock02-033.png', 'rock07-k03-039.png', 'rock06ck02-055.png', 'rock04-071.png', 'rock02-087.png', 'rock06ck02-065.png', 'rock02-004.png', 'rock03-093.png', 'rock02-006.png']
['paper02-035.png', 'paper05-074.png', 'paper03-030.png', 'paper06-058.png', 'paper03-054.png', 'paper04-041.png', 'paper01-109.png', 'paper06-107.png', 'paper02-114.png', 'paper07-044.png']
['testscissors02-114.png', 'scissors01-000.png', 'testscissors01-041.png', 'testscissors03-051.png', 'testscissors03-106.png', 'scissors03-037.png', 'testscissors02-033.png', 'testscissors02-024.png', 'scissors03-034.png', 'scissors04-115.png']
```  

### 各打印两张石头剪刀布训练集图片  
```
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```  
<img src="pic\p1.png"  width = "35%" /><img src="pic\p2.png"  width = "35%" />
<img src="pic\p3.png"  width = "35%" />  <img src="pic\p4.png"  width = "35%" />
<img src="pic\p5.png"  width = "35%" /><img src="pic\p6.png"  width = "35%" />

### 调用TensorFlow的keras进行数据模型的训练和评估。Keras是开源人工神经网络库，TensorFlow集成了keras的调用接口，可以方便的使用。  

```
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")
```  
  
  
```
2023-06-07 04:12:07.367472: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-06-07 04:12:07.367530: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
Found 2520 images belonging to 3 classes.
Found 372 images belonging to 3 classes.
2023-06-07 04:12:26.757627: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2023-06-07 04:12:26.757683: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-06-07 04:12:26.757713: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-a62658): /proc/driver/nvidia/version does not exist
2023-06-07 04:12:26.758755: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 6272)              0         
...
Total params: 3,473,475
Trainable params: 3,473,475
Non-trainable params: 0
_________________________________________________________________
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
2023-06-07 04:12:30.017383: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.
Epoch 1/25
2023-06-07 04:12:35.034346: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 34020000 exceeds 10% of free system memory.
2023-06-07 04:12:35.057941: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 706535424 exceeds 10% of free system memory.
2023-06-07 04:12:36.311879: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 176633856 exceeds 10% of free system memory.
2023-06-07 04:12:36.403599: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 167215104 exceeds 10% of free system memory.
20/20 [==============================] - 87s 4s/step - loss: 1.2325 - accuracy: 0.3433 - val_loss: 1.0895 - val_accuracy: 0.3333
Epoch 2/25
20/20 [==============================] - 78s 4s/step - loss: 1.0640 - accuracy: 0.4206 - val_loss: 0.8220 - val_accuracy: 0.8145
Epoch 3/25
20/20 [==============================] - 78s 4s/step - loss: 0.9519 - accuracy: 0.5254 - val_loss: 0.6328 - val_accuracy: 0.6855
Epoch 4/25
20/20 [==============================] - 76s 4s/step - loss: 0.6937 - accuracy: 0.6857 - val_loss: 0.8841 - val_accuracy: 0.5699
Epoch 5/25
20/20 [==============================] - 76s 4s/step - loss: 0.6703 - accuracy: 0.7040 - val_loss: 0.6849 - val_accuracy: 0.6129
Epoch 6/25
20/20 [==============================] - 77s 4s/step - loss: 0.5466 - accuracy: 0.7750 - val_loss: 0.1347 - val_accuracy: 1.0000
Epoch 7/25
20/20 [==============================] - 79s 4s/step - loss: 0.4059 - accuracy: 0.8262 - val_loss: 0.8982 - val_accuracy: 0.6505
Epoch 8/25
20/20 [==============================] - 76s 4s/step - loss: 0.4067 - accuracy: 0.8433 - val_loss: 0.2464 - val_accuracy: 0.9247
Epoch 9/25
20/20 [==============================] - 75s 4s/step - loss: 0.4207 - accuracy: 0.8500 - val_loss: 0.0740 - val_accuracy: 0.9758
Epoch 10/25
20/20 [==============================] - 74s 4s/step - loss: 0.2586 - accuracy: 0.9000 - val_loss: 0.0719 - val_accuracy: 0.9731
Epoch 11/25
20/20 [==============================] - 75s 4s/step - loss: 0.4036 - accuracy: 0.8730 - val_loss: 0.0516 - val_accuracy: 1.0000
Epoch 12/25
20/20 [==============================] - 74s 4s/step - loss: 0.1765 - accuracy: 0.9349 - val_loss: 0.0658 - val_accuracy: 0.9866
Epoch 13/25
20/20 [==============================] - 77s 4s/step - loss: 0.2615 - accuracy: 0.9016 - val_loss: 0.0440 - val_accuracy: 0.9785
...
Epoch 24/25
20/20 [==============================] - 73s 4s/step - loss: 0.1438 - accuracy: 0.9587 - val_loss: 0.0506 - val_accuracy: 0.9758
Epoch 25/25
20/20 [==============================] - 75s 4s/step - loss: 0.1018 - accuracy: 0.9647 - val_loss: 0.0566 - val_accuracy: 0.9704
```

ImageDataGenerator是Keras中图像预处理的类，经过预处理使得后续的训练更加准确。  


Sequential定义了序列化的神经网络，封装了神经网络的结构，有一组输入和一组输出。可以定义多个神经层，各层之间按照先后顺序堆叠，前一层的输出就是后一层的输入，通过多个层的堆叠，构建出神经网络。  


神经网络两个常用的操作：卷积和池化。由于图片中可能包含干扰或者弱信息，使用卷积处理（此处的Conv2D函数）使得我们能够找到特定的局部图像特征（如边缘）。此处使用了3X3的滤波器（通常称为垂直索伯滤波器）。而池化（此处的MaxPooling2D）的作用是降低采样，因为卷积层输出中包含很多冗余信息。池化通过减小输入的大小降低输出值的数量。详细的信息可以参考知乎回答“如何理解卷积神经网络（CNN）中的卷积和池化？”。更多的卷积算法参考Github Convolution arithmetic。  


Dense的操作即全连接层操作，本质就是由一个特征空间线性变换到另一个特征空间。Dense层的目的是将前面提取的特征，在dense经过非线性变化，提取这些特征之间的关联，最后映射到输出空间上。Dense这里作为输出层。  


### 完成模型训练之后，我们绘制训练和验证结果的相关信息。

  
```
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
```  

<img src="pic\p7.png"  width = "100%" />
  









