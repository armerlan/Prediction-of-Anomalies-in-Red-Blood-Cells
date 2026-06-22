## Introduction

This project has been submitted to the Laboratory of Information Photonics and Optical Metrology, Department of Physics, Indian Institute of Technology (BHU), Varanasi for the award of the Integrated Dual Degree in Engineering Physics.

This repository provides an end-to-end user-oriented framework for AI-assisted quantitative phase imaging, enabling automated segmentation, feature extraction, and anomaly prediction in red blood cells.

## Description of Folders

1. The 'training' folder contains files associated with the generation of the dataset (`generate_dataset.py`) and then training and comparison of models (`analyze_data.ipynb`). The trained model, features used, threshold and other metadata are saved to the 'models' folder through the `analyze_data.ipynb` file.

2. The 'backend' folder consists of the main functions that are used to process the data in the application. The `pipeline.py` file is the connection between the 'application' folder and the 'backend'.

3. The 'application' folder contains the files associated with various tabs within the application, along with the dialog boxes and logger. The `worker.py` runs computationally intensive tasks (model loading, hologram reconstruction, and image analysis) in background threads, communicating with the GUI via signals for progress updates, completion, errors, and cancellation, so that the PyQt interface never freezes during processing.

## Running the Application

Before running the application, please make sure the data folders have been uploaded as needed and that the .env file is ready (if required).

### Through the Terminal

The application can be run through the terminal by activating the virtual environment inside the directory and then executing,

`python -m application.main`

### Through the .exe file

To convert the code into an executable file, run the following command all at once inside the terminal (the name QuantPhase can be changed as appropriate)

```bash
pyinstaller --onefile ^
--windowed ^
--name QuantPhase ^
--add-data "application/about.md;application" ^
--add-data "application/assets;application/assets" ^
--add-data "models;models" ^
--collect-all xgboost ^
--add-data "models/cellpose;models/cellpose" ^
application/main.py
```

This will save the .exe to a new folder called 'build'. The first time the application opens, it might take a couple of minutes due to system checks.

> [!NOTE]
> 1. The segmentation process takes a couple of minutes to complete, during which the progress bar remains at 30% because it was challenging to obtain progress information while the Cellpose model runs in the background.
> 2. The better the reconstruction, the better the segmentation and prediction results.

## Copyright

Copyright © Indian Institute of Technology (BHU), Varanasi.

This software was developed by Alankrita Parmeshwar under the guidance of Dr. Rakesh K. Singh (Professor, Department of Physics, IIT BHU, Varanasi) as part of an Integrated Dual Degree (B.Tech. + M.Tech.) thesis in Engineering Physics at IIT (BHU), Varanasi.

The author is permitted to reproduce and authorize reproduction of derivative works, provided that the source and the Institute's copyright notice are indicated.
