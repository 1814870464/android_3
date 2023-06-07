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
![p1]()  
![p2]()  
![p3]()  
![p4]()  
![p5]()  
![p6]()  


  
  
  









