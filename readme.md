# YOLO-CNN-LSTM Model for Image Classification
1.项目标题与内容概述

标题：深度学习用于检测电影和连续剧中的暴力和虐待行为： OTT 流媒体平台
概述：本项目利用深度学习和计算机视觉技术检测并移除电影视频中的暴力和不当内容。与目前常见的仅识别是否为暴力的检测方法不同的是，该框架还对识别到的暴力行为进行初级暴力和高级暴力的分类。它结合了YOLO和CNN-LSTM模型，能够以70%的准确率检测和分类暴力内容。随着人们对好莱坞影片暴力内容的关注增加，该框架填补了当前流媒体平台内容审核的空白，减少了对人工内容审核的依赖，提高了效率。它在不同视频类型上的可扩展性使其成为数字平台内容管理的有力工具。

1. Project title and content overview
 
Title: Deep learning for violence and abuse detection in cinema and series: OTT streaming platforms
Content: This project uses deep learning and computer vision techniques to detect and remove violent and inappropriate content from movie videos. Unlike current common detection methods that only identify whether the violence is violent or not, the framework also categorizes the identified violence in terms of lowand high levels  in movies. It combines YOLO and CNN-LSTM models to detect and categorize violent content with 70% accuracy. With the increased concern about violent content in Hollywood movies, the framework fills the current gap in content review for streaming platforms, reducing the reliance on manual content review and improving efficiency. Its scalability across different video types makes it a powerful tool for content management on digital platforms.

## Project Structure
2.数据集，预处理及项目结构
数据集AIRTLab：https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos
我们对上述数据集做了新的整理分类，划分为非暴力（包括手势，握手，击掌，拥抱，跳跃，行走，打招呼），初级暴力（包括推搡，扇耳光，窒息，打架，踢，拳击），高级暴力（包括枪击，刺伤，棍棒），数据结构如下：
dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
        class3/
            img1.jpg
            img2.jpg
            ...
    test/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
         class3/
            img1.jpg
            img2.jpg
            ...


- `cnn.ipynb`: Jupyter Notebook containing the CNN model implementation, training, and evaluation.

## Requirements

To run the code in this repository, you need to have the following libraries installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Jupyter

You can install the required packages using the following command:

```bash
pip install tensorflow keras numpy matplotlib opencv-python-headless jupyter
```

Dataset
The dataset used for this project should be structured in the following format:
```
dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
    test/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
``` 
  
git clone https://github.com/yourusername/cnn-image-classification.git
cd cnn-image-classification

jupyter notebook cnn.ipynb
