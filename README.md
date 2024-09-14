# DEPLOY ML PIPELINE WITH FASTAPI

## PROJECT DESCRIPTION
The project in the udacity course is to build an AI service with full CI/CD for a machine learning model deployment process. In this project, I learned how to implement github actions, write test cases for the API, and deploy on RENDER.com.

### FOLDER OF PROJECT "STARTER"
```bash
.
├── data
│   └── census.csv
├── main.py
├── model
│   └── model.pkl
├── README.md
├── requirements.txt
├── sanitycheck.py
├── screenshots
│   ├── continous_deloyment.png
│   ├── continous_integration.png
│   ├── live_get.png
│   |── live_post.png
|   └── output_training.png
├── setup.py
├── starter
│   ├── __init__.py
│   ├── ml
│   │   ├── data.py
│   │   ├── __init__.py
│   │   ├── model.py
│   ├── model_card_template.md
│   ├── predict_model.py
│   ├── src
│   │   ├── helper_utils.py
│   └── train_model.py
└── test_api.py

```
### ENV SET UP
Set up with anaconda

```bash
conda create --name env python=3.11
conda activate env
```
After that, you must install all python packages:
```bash
pip install -r starter/requirements.txt
```
Move on project folder
```bash
cd starter
```
### DATA
The data in data folder : data/census.csv
### TRAINING
You can preprocess data and training model in starter folder:

The MODEL_CARD_TEMPLATE: [Model_card](model_card_template.md)

```bash
python train_model.py
```
After that, the output slide : slice_output.txt

### SERVING AI MODEL WITH FASTAPI
You can testing in local : ```uvicorn main:ai --reload ```

### UNIT TEST

```bash
python test_api.py
```
and 
```bash
python sanitycheck.py
```

### CI/CD
Set up the manual.yaml :
```bash
name: CI/CD Pipeline

on:
  push:
    branches:
      - master
  pull_request:
    branches:
      - master

jobs:
  build:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: [3.11]
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r starter/requirements.txt
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 --ignore=E501,F401,W503,W391,E266,W293 .
    - name: Test with pytest
      run: |
        pip install pytest
        pytest -v    
    - name: Test with santity check
      run: |
        python starter/sanity_check.py"
```
Push code and monitoring in github actions

### DEPLOY WITH RENDER

Create the account in link : https://dashboard.render.com/web

Here is the web deployed: https://deploy-ml-pipeline-in-production.onrender.com/docs
