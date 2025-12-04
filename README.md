# Roboflow model
This repo contains python code used to locally run a YOLOv8 object detection and instance segmentation model trained from [Roboflow.com](https://app.roboflow.com/mastersthesis-d11wq)

All images used for training, testing, and validation were collected as part of my masters thesis. This model is trained to recognize the instrument aperture of an LI-6800 photosynthesis aperature, identify, and count needles contained within the aperature, and calculate a ratio of instrument to leaf pixels to correct leaf area measurements. 

## Files
| **Folder**     | **Contents**     |**Function**   |
| ------------- | ------------- | ------------- |
| code | Workflow runner, annotation downloader, and model evaluation .py and .md files | Files used to run the model locally |
| train | Annotated images imported from Roboflow | Used to train model |
| test | Annotated images imported from Roboflow | Used for performance evaluation | 
| valid | Annotated images imported from Roboflow | Used for performance evaluation |
| ground_truth | Annotations from test, train, and valid sets compiled into a .JSON file | Annotated images used by `model_evaluation.py` to obtain performance metrics |
| results | Results folder containing graphs and text files of performance metrics and full model .csv output | Visualize model performance for tuning and data extrapolation |

## Running and evaluating the model
Instructions to run the Roboflow model are contained in `workflow_ReadMe.md`

Instructions for model evaluation are contained in `model_evaluation_ReadMe.md`
