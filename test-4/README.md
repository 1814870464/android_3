# 实验4  
- 使用[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)训练自定义的图像分类器
- 利用Android Studio导入训练后的模型，并结合CameraX使用
- 利用手机GPU加速模型运行  

## 下载并运行初始代码，创建工作目录  
```
git clone https://github.com/hoitab/TFLClassify.git
```  
## 连接物理机
- 手机通过USB接口连接开发平台，并设置手机开发者选项允许调试。
- 选择真实物理机（而不是模拟器）运行start模块
- 允许应用获取手机摄像头的权限，得到下述效果图，界面利用随机数表示虚拟的识别结果。  

## 向应用中添加TensorFlow Lite
* 右键“start”模块，或者选择File，然后New>Other>TensorFlow Lite Model
* 选择已经下载的自定义的训练模型。本教程模型训练任务以后完成，这里选择finish模块中ml文件下的FlowerModel.tflite。
* 点击“Finish”完成模型导入，系统将自动下载模型的依赖包并将依赖项添加至模块的build.gradle文件。

最终TensorFlow Lite模型被成功导入，并生成摘要信息  
![模型导入](https://github.com/1814870464/android_3/blob/main/test-4/picture/p1.png)

## 添加代码重新运行APP  
定位“start”模块MainActivity.kt文件的TODO 1
![模型代码](https://github.com/1814870464/android_3/blob/main/test-4/picture/p2.png)  

### 添加代码
```kotlin
 private class ImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :
        ImageAnalysis.Analyzer {

        // TODO 1: Add class variable TensorFlow Lite Model
        private val flowerModel = FlowerModel.newInstance(ctx)
        // Initializing the flowerModel by lazy so that it runs in the same thread when the process
        // method is called.

        // TODO 6. Optional GPU acceleration


        override fun analyze(imageProxy: ImageProxy) {

            val items = mutableListOf<Recognition>()

            // TODO 2: Convert Image to Bitmap then to TensorImage
            val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))
            // TODO 3: Process the image using the trained model, sort and pick out the top results
            val outputs = flowerModel.process(tfImage)
                .probabilityAsCategoryList.apply {
                    sortByDescending { it.score } // Sort with highest confidence first
                }.take(MAX_RESULT_DISPLAY) // take the top results
            // TODO 4: Converting the top probability items into a list of recognitions
            for (output in outputs) {
                items.add(Recognition(output.label, output.score))
            }
            // START - Placeholder code at the start of the codelab. Comment this block of code out.
//            for (i in 0..MAX_RESULT_DISPLAY-1){
//                items.add(Recognition("Fake label $i", Random.nextFloat()))
//            }
            // END - Placeholder code at the start of the codelab. Comment this block of code out.


            // Return the result
            listener(items.toList())

            // Close the image,this tells CameraX to feed the next image to the analyzer
            imageProxy.close()
        }

        /**
         * Convert Image Proxy to Bitmap
         */
        private val yuvToRgbConverter = YuvToRgbConverter(ctx)
        private lateinit var bitmapBuffer: Bitmap
        private lateinit var rotationMatrix: Matrix

        @SuppressLint("UnsafeExperimentalUsageError")
        private fun toBitmap(imageProxy: ImageProxy): Bitmap? {

            val image = imageProxy.image ?: return null

            // Initialise Buffer
            if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                Log.d(TAG, "Initalise toBitmap()")
                rotationMatrix = Matrix()
                rotationMatrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                bitmapBuffer = Bitmap.createBitmap(
                    imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
                )
            }

            // Pass image to an image analyser
            yuvToRgbConverter.yuvToRgb(image, bitmapBuffer)

            // Create the Bitmap in the correct orientation
            return Bitmap.createBitmap(
                bitmapBuffer,
                0,
                0,
                bitmapBuffer.width,
                bitmapBuffer.height,
                rotationMatrix,
                false
            )
        }

    }
```   


## 运行效果
### 扫描郁金香  
<img src="picture\p3-1.jpg" width="35%"/>  

### 扫描玫瑰  
<img src="picture\p3-2.jpg" width="35%"/>  

### 扫描蒲公英  
<img src="picture\p3-3.jpg" width="35%"/>  
