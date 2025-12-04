# Roboflow model
This repo contains python code used to locally run a YOLOv8 object detection and instance segmentation model trained from [Roboflow.com](https://app.roboflow.com/mastersthesis-d11wq)

## Files
| **Folder**     | **Contents**     |**Function**   |
| ------------- | ------------- | ------------- |
| code | Workflow runner, annotation downloader, and model evaluation .py and .md files | Files used to run the model locally |
| ground_truth | Test, training, and validation images | Annotated images used by `model_evaluation.py` to obtain performance metrics |
| results | Results folder containing graphs and text files of performance metrics and full model .csv output | Visualize model performance for tuning and data extrapolation |

