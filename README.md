
# Pokemon MLOps Project Repository
[![Build Pipeline](https://github.com/alaeddine245/pokemon-mlops-project/actions/workflows/build.yaml/badge.svg)](https://github.com/alaeddine245/pokemon-mlops-project/actions/workflows/build.yaml)

![image](https://github.com/alaeddine245/pokemon-mlops-project/assets/51765101/74ecd110-2940-435b-9a12-78c6be55d8fb)

This repository contains the code, documentation and necessary resources for a Pokemon project with MLOps. The repository is structured as follows:

```
├── .github
│   └── workflows
│       └── build.yaml
├── data
│   └── download.py
├── docs
│   ├── .gitkeep
│   └── README.md
├── models
│   └── .gitkeep
├── notebooks
│   ├── .gitkeep
│   └── pokemon_generation_one_classification.ipynb
├── src
│ 	└── constants.py
│   └── training_pipeline.py
│   └── mlflow_setup.py
│   ├── evaluation
│   │   └── evaluate.py
│   ├── models
│   │   └── model.py
│   ├── prediction
│   │   └── predict.py
│   ├── preprocessing
│   │   └── preprocess_split.py
│   ├── training
│   │   └── train.py
│   ├── visualization
│   │   ├── __init__.py
│   │   ├── plot_results.py
│   └── __init__.py
├── tests
│   ├── test_preprocess.py
│   └── test_train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

``` 

## `.github`

This directory contains the CI/CD workflow definition for the project, located in the `workflows` directory.

### `build.yaml`

This file defines the build pipeline for the project.

## `data`

This directory contains the data used for the project.

### `download.py`

This file contains the script for downloading the data for the project.

## `notebooks`

This directory contains the Jupyter notebooks used for experimentation and exploration.

### `.gitkeep`

This file is used to ensure that the `notebooks` directory is not empty and is tracked by Git.

### `pokemon_generation_one_classification.ipynb`

This Jupyter notebook contains the code for the machine learning pipeline for classifying Pokemon from the first generation.

## `src`

This directory contains the source code for the project.

### `evaluation`

This directory contains the script for evaluating the trained model.

### `models`

This directory contains the script for defining the machine learning model.

### `prediction`

This directory contains the script for making predictions using the trained model.

### `preprocessing`

This directory contains the script for preprocessing the data.

### `training`

This directory contains the script for training the machine learning model.

## Run The Project

 1.  Download the data using data/download.py or directly through this [link](https://www.kaggle.com/datasets/bhawks/pokemon-generation-one-22k) on Kaggle


 2. This project is using MLflow which hosted on Azure: http://mlops.uksouth.cloudapp.azure.com:5000
launch mlflow server
	```
	mlflow server -h 0.0.0.0
	```
 3. Launch training 
	```
	cd pokemon-mlops-project-main/src/
	python training_pipeline.py
	```
	The training pipeline will start and you can see the execution, metrics and different information in the mlflow UI
  <img width="1502" alt="image" src="https://github.com/alaeddine245/pokemon-mlops-project/assets/51765101/25b84290-3100-4b01-b071-4a913cb1dbde">
  <img width="1502" alt="image" src="https://github.com/alaeddine245/pokemon-mlops-project/assets/51765101/cece99e2-81d8-4a78-ba86-74f0c3313667">

  
 4. The training pipeline can be also launched with opentelemetry and you can inspect the different logs about different metrics, cpu usage, information about the file, etc.. when the model is training 
	```
	opentelemetry-bootstrap -a install
	cd pokemon-mlops-project-main/src/

	opentelemetry-instrument --traces_exporter console --metrics_exporter console python pokemon-mlops-project-main/src/training_pipeline.py > output
	```
  The logs outputed by by opentelemetry can be seen in OToutput.txt file
## Tests
[![Build Pipeline](https://github.com/alaeddine245/pokemon-mlops-project/actions/workflows/build.yaml/badge.svg)](https://github.com/alaeddine245/pokemon-mlops-project/actions/workflows/build.yaml)	
- The `tests` directory contains the tests for the different scripts in the project which are automated as part of build pipeline and run when pushing to the main branch
- The build pipeline also containe pylint command to analyse the code and give suggestions on improving the code qualities

## Improvment TODO

 - [ ] Add more unit tests
 - [ ] Create deployment pipeline
 - [ ] Integrate the training pipeline as part of CI pipeline
 - [ ] Create CD pipeline in github action
 - [ ] Integrate deployment pipeline as part of CD pipeline
 - [ ] Add different environments for the project (Test/Prod)
 
