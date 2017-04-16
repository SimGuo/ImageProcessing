# ImageProcessing Project

---
> 把Imagedata文件名都转换成英文，把原先的图像数据集里面不规范的文件名都修改规范了，每个类别里面的sift文件夹，是我提取出来的sift文件

*main.cpp:*

1. 使用一部分数据作为训练集，对其中每张图片提取SIFT特征。
2. 将所有提取出来的SIFT特征混在一起，用kmeans将其分为K类，也就是K个词汇。
3. 用生成的词汇集去处理原先的图片集合，对原先的每张图片重新计算出一个K维的特征向量。每个维度是词汇集中词汇出现的次数。
4. 用生成的K维特征向量，结合labels成为训练集，用SVM进行训练。
5. 这里SVM我用了五个分类器来达到多分类的效果

*parameters & function:*

  - `mySVM[5]:` 设置了五层分类器

  - `PresetLabels:`我假想的SVM是个二叉树形的判别器，所以为每个节点预设了被五层判别器被判别的结果

  - `WriteData:` 将Mat数据写入相应文件



