# PneunomiaDetection

### Author
Lukas Benner (...)

Kevin Gavagan (...)

Lea Soffel (4962704)

## Abstract


## Introduction
Pneumonia is one of the most fatal infectious disease. In Germany around 400,000 to 600,000 people die per year because of Pneumonia. Especially for children under 5 it is the most fatal infection. Pneumonia is caused by microorganisms, which can be visualized by imaging methods as X-ray or CT. Symptoms of Pneumonia are cough, fever and dyspnea.
The goal of this work is to generate an AI which detects Pneumonia on a X-ray thorax. Input is an grayscale imgae (224x224) of a X-ray thorax, with the help of a neural network from scratch the algorithm outputs a 0 or a 1. 0 means no pneumonia, where as 1 means pneumonia is detected. 


## Related Work
Our work is based on a data set found on kaggle [2]. The data set is used in a lot of algorithms found on kaggle. Another project was found from University of California San Diego [3].


## Dataset and Features
Our work is based on a data set found on kaggle [2]. The dataset contains 5,856 labeled X-ray thorax images (JPEG). These iamges are split into to sets, a training set and a test set. Each set is divided into two sets, NORMAL and PNEUNOMIA. For training we have 5,232 images divided into 1,349 NORMAL and 3,883 PNEUMONIA. To test the algorithm we have 624 images divided into 234 NORMAL and 390 PNEUMONIA. 
|                 | Total | Training set | Test set |
| --------------- | ----- | ------------ | -------- |
|Number of Images | 5,856 | 5,232        |  624     |
|Percentage       | 100%  | 89,34%       | 10,66%   |

The original image size varies and is extremely big, so we wrote a python script to down scale all images to size 224x224. You can find the python script in the repository (PythonImageScaler.py). 
Because of the fact that we use grayscale images we only need one color attribute per pixel, instead of three (RGB).
The X-ray images are from patients age one to five from Guangzhou Women and Children's Medical Center [4].
Example for a NORMAL X-ray thorax:

![NORMAL](/scaled_chest_xray/test/NORMAL/NORMAL-1049278-0001.jpeg)

Example for a PNEUMONIA X-ray thorax:

![PNEUMONIA](/scaled_chest_xray/test/PNEUMONIA/BACTERIA-1135262-0004.jpeg)


HIER VIELLEICHT NOCH WAS ZU LOAD DATA

## Methods
We wrote our algorithm from scratch. Its a neural network 


## Experiments/Results/Discussion



## Conclusion


## References
https://www.lungeninformationsdienst.de/krankheiten/lungenentzuendung/verbreitung
[2] https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
[3] https://data.mendeley.com/datasets/rscbjbr9sj/2
[4] https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
