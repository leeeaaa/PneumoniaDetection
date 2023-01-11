# Pneumonia Detection

## How to setup

1. Create am Anaconda environment
2. Install the requirements with `pip install -r requirements.txt`


## How to use

- There are two Jupyter Notebooks in the src folder.
- Open the Notebooks, go through the code and train the models.

**DISCLAIMER**
Training the models can take up to 30 min.
If you don't have that much time, decrease the number of iterations or epochs.
Be aware that the resulting model will be not as good.


## Files and Folders

### Files

| Filename | Description |
|----------|-------------|
| src\PneumoniaDetection.ipynb | The notebook with the neural network from scratch |
| src\TensorflowPneunomialDetection.ipynb | The notebook with the NN and CNN implemented with Tensorflow. |
| src\PythonImageScaler.py | Script to prepare the image dataset |
| documentation\main.pdf | The documentation |


### Folders

| Foldername | Description |
|----------|-------------|
| documentation | Documentation of the project |
| src\scaled_chest_xray | Dataset scaled to 224x224 |
| src\scaled2_chest_xray | Dataset scaled to 56x56 |
