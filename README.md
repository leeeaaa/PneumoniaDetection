# PneunomiaDetection

### Author
Lukas Benner (3277496)

Kevin Gavagan (2240332)

Lea Soffel (4962704)

## Abstract
In this article we describe the development of 3 algorithms to detect Pneumonia disease on X-ray thorax images. Pneumonia is one of the most fatal infections worldwide, especially for young people under 5 and people over 65 the disease often ends deadly. Because young people dont show any signs, imaging methods such as X-ray or CT are used to detect the disease. In the past years the topic of using an AI to detect diseases is becoming an increasingly important issue. So why not let an AI do it for you?

## Introduction
Pneumonia is an inflammation of the lung tissue especially the air sacs in one or both lungs. Symptoms of Pneumonia are cough, shortness of breath, chest pain, fever and dyspnea. The symptomps vary depending on age, health problems, weakened immune system and type of germ, which causes the infection. For example young people often dont show any symtomps at all. Pneumonia is one of the most fatal infectious disease worldwide. In Germany around 400,000 to 600,000 people die per year because of Pneumonia disease. Especially for children under 5 it is the most fatal infection, killing more young people than maleria, AIDS and other diseases. Pneumonia is caused by microorganisms such as bacteria, viruses and funghi, which can be visualized by imaging methods as X-ray or CT. Pneumonia can have a mild to life-threatening course. Bacterial and viral agents are the popular causes but differ in their treatment methods. Bacterial pneumonia needs to be treated with antibiotics, whereas viral pneumonia is treated with antivirals and other supportive care. Therefore it is important to detect pneumonia in an early state to combat the bacteria or virus. 
Because of the fact that newborns often dont show any signs of the disease the possibility of detecting Pneumonia by imaging methods such as X-ray or CT is an important medical method. Additionally, X-ray is a standard treatment and can help differentiate between the different cause for the infection [1-5].

The image below shows X-ray thorax images of a person without pneumonia and a viral and a bacterial caused pneumonia [5].

![TypesOfPneumonia](typesofpneumonia.jpg)

The goal of this work is to generate an AI which detects Pneumonia on a X-ray thorax image. Input is an grayscale imgae (224x224) of a X-ray thorax, with the help of a neural network, which we developed the algorithm outputs a 0 or a 1. 0 means no pneumonia, where as 1 means pneumonia is detected. In the future the use of the algorithm could be to support or validate the decision of doctors. We trained and tested the model with data of young people age 5 and less because they often show no symptomps.


## Related Work
Our work is based on a data set found on kaggle [2]. The data set is used in a lot of algorithms found on kaggle. The Approaches are very different, often they use CNN as network. Another project was found from University of California San Diego [3].
HIER MUSS NOCH MEHR CONTEXT HIN


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

Data augmentation is not included in the model because X-ray scans are only taken in a specific orientation, and variations such as flips and rotations will not exist in real X-ray images.

HIER VIELLEICHT NOCH WAS ZU LOAD DATA

## Methods
We decided to implement the algorithm in three ways, the input and the output is always the same. 

One algorithm is a neural network completly from scratch. We wanted to be as flexible as possible, which is why we also implemented the possibility to choose different amount and size of the hidden layers. Also the hyperparameters lambd, learning rate and number of iterations are adjustable. What we did not implement is the training of the hyperparameters.

cost function: ?

hyperparameter: ?

architecture: ?

The second algorithm is also a neural network but it uses Tensorflow. With the help of keras we load the data, create a model and predict the output. 

loss function: ?

hyperparameter: ?

architecture: ?


The third algorithm is a convolutional neural network.

cost function: ?

hyperparameter: ?

architecture: ?


The reason why we decided to implement all three methods is that we wanted to compare the efficeincy, the accuracy and the computational effort of these three. More about this later.


## Experiments/Results/Discussion
HIER DIE ACCURACIES UND UND UND


## Conclusion


## References
[1] https://www.lungeninformationsdienst.de/krankheiten/lungenentzuendung/verbreitung
[2] https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
[3] https://data.mendeley.com/datasets/rscbjbr9sj/2
[4] https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
[5] https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
