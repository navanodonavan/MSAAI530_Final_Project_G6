# Project Title : Predicting Power Generation Impacts from Weather Changes with Deep Learning and Time Series Models

This project is a part of the AAI-530 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

### Project Status: [In Progress]

## Installation

TBD

Launch Jupyter notebook and open the `TBD.ipynb` file from this repository. 

The `TBD.ipynb` is the final version.

## Required libraries to be installed including:

    import math
    import os
    import librosa
    import librosa.display
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from collections import Counter

  
## Project Intro/Objective

We are utilizing the Sunlab Faro PV plant data and the Sunlab Faro Meteo data (https://www.edp.com/en/innovation/open-data/data) to predict solar power generation based on weather changes using deep learning and time series models.

### Partner(s)/Contributor(s)

•	Donavan Trigg

•	Daniel Shifrin

•	Michael Skirvin


### Methods Used

•	Time Series

•	Machine Learning

•	Neural Networks

•	Deep Learning


### Technologies

•	Python

•	Jupyter Notebook

•	PyTorch


### Project Description

We've combined the Solar Plant data and the Meteo data together from 2014, 2015, 2016, and 2017 for our modeling. We've cleaned the data for missing values and removed the Horizontal and Vertical panel orientations. We are focusing on the "Optimal" panel orientation for our analysis.
