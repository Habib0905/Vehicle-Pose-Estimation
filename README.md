# Vehicle Pose Estimation for Autonomous Driving Using YOLOv8

This project focuses on enhancing vehicle pose estimation for autonomous driving systems using the **YOLOv8** deep learning model. The model is fine-tuned and trained on the **CarFusion** dataset using about **30,000** images. It was further improved with a custom dataset that captures unique traffic conditions in various regions of Bangladesh.

Authors:
**Tanzir Razzaque**, **Habib Hussain**, **Rafid Ahmmad**\n
{tanzir.razzaque, habib.hussain, rafid.ahmmad}@northsouth.edu

## Key Features

### 🚗 Vehicle Pose Estimation
Accurate detection of vehicle orientation and position for better navigation and collision avoidance.

### 📊 Custom Dataset
A dataset tailored for the diverse and dynamic traffic scenarios in Bangladesh, improving the model's performance in real-world applications.

### 🔧 Hyperparameter Tuning
Pre-training optimization to achieve high accuracy and real-time performance.

### ⚡ Performance
**YOLOv8** outperforms traditional methods in terms of accuracy and computational efficiency, making it a robust solution for vehicle pose estimation in autonomous driving systems.

---

This project demonstrates the vast potential of **YOLOv8** in addressing the challenges of real-time pose estimation in complex traffic environments.

 



## :hammer_and_wrench: Setup :
## Dependencies

### Install Ultralytics
```sh
pip install ultralytics
```

### Import YOLO

```sh
from ultralytics import YOLO
```


## Utilize our model for training and testing

### Dataset
Our dataset was in COCO format, and to convert it to YOLO format we used our custom COCO to YOLO converter.
If your annotated dataset is in COCO format, you can use the converter given in this repository.

Your dataset should be in the following folder structure
```
YOLOv8-Dataset/
│
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
│
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   └── val/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...

```
### Download the weights of the model from below
[Weights](https://drive.google.com/drive/folders/17u0B0aKTYkY8I72gQLl2tsEyvQzDavJp?usp=sharing)

There are two weight files in the link, best.pt and last.pt. In our experiment the last.pt was found to be the more robust one.

### Hyperparameter Tuning

Follow the code, 'car_pose_tuning_training.ipynb' given in this repository.

```sh
model = YOLO('/path/to/weight.pt')
model.tune(data='/path/to/config.yaml', epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
     
```
Input the path to the config file and the weight you downloaded.

The config file is provided in the repository.
Example:
```sh
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /path/to/dataset  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images


# Keypoints
kpt_shape: [14, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
# flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes dictionary
names:
  0: car
     
```
### Training
Follow the code file, 'car_pose_tuning_training.ipynb' given in this repository.
```sh
model = YOLO("/path/to/weight.pt")
model.train(data = '/path/to/config.yaml', cfg='/path/to/best_hyperparameters.yaml', epochs =100, imgsz = 640)
  
```
Input the path to the previous config file and the path to the weight you downloaded.
In 'cfg', input the path to the config file which you will be provided after hyperparameter tuning.


### Training without Hyperparamer Tuning

If you want to just train your dataset using our model, you can proceed like below:

```sh
model = YOLO("/path/to/weight.pt")
model.train(data = '/path/to/config.yaml', epochs =100, imgsz = 640)
  
```
Input the path to the config file and the weight you downloaded.


### Testing
Follow the code file, 'car_pose_inference.ipynb' given in this repository.

#### For inference with images

```sh
folder_path = '/path/to/testImages'
model_path = '/path/to/weight.pt'
```
Input the paths to your test images and your weight.

#### For inference with video
```sh
source =  "/path/to/video.MP4"
```
Input the path to your test video.

#### For inference with web camera
Follow the code file, 'webcam_inference.py' and run on your local machine.
```sh
model = YOLO('/path/to/weight.pt')
```
Input the path to the weight.
