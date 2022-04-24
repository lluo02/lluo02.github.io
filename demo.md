layout: page
title: "Project Demo"
permalink: /demo

# Demo

The problem this dataset presents is training a model, based on the performance of an individual in employee training, whether or not the said employee is likely to stay with the company or look for a new job. The dataset is imbalanced with about a 3:1 majority of 0 (Not looking for new job) to 1 (Looking for new job) entries.

Data: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists?select=aug_test.csv

The data are preprocessed by standard normalization of columns, separation into training, validation, and test split, and the replacment of null values with the mode of their respective columns.

## **Virtual Environment**
This project is built into a virtual environment using WSL 2.0 on Ubuntu 20.04.3. To build the virtual environment, the provided Pipfile contains the relevant dependencies. To build from scratch, do  

> pip install pipenv
> pipenv install numpy pandas scikit-learn xgb flask gunicorn

To activate the virtual environment afterwards, run 

> pipenv shell

This project can also be deployed to a Docker container running on a Ubuntu 20.04.3 image using WSL 2.0. In order to build it, navigate to the directory that contains the Dockerfile and run

> docker build -t enrollee-predict .

To deploy the Docker container after it has been build, run

> docker run -it --rm -p 9696:9696 enrollee-predict

The project will run on **localhost:9696** once deployed.

## About the model

The model is created using XGBoost. There are three parameters I tune in the model's creation: 
- The number of iterations to train over
- Learning rate
- Max depth of a tree
- Min weight of a child

The longer the model attempts to train itself for, the more likely it overfits. That is, the model adjusts itself too much to the training data that it can't measure anything else.
The same thing can happen if the learning rate is too high, the trees are too deep, or if the weight of the children is too small.
The goal is to maximimize the performance of the model while avoiding extremes.

## Using the model

Let's take a look at this sample employee.
```
enrollee = {
    "enrollee_id": 11674,
    "city": "city_83",
    "city_development_index": 0.923,
    "gender": "male",
    "relevent_experience": "has_relevent_experience",
    "enrolled_university": "no_enrollment",
    "education_level": "masters",
    "major_discipline": "stem",
    "experience": "12",
    "company_size": "1000-4999",
    "company_type": "pvt_ltd",
    "last_new_job": "2",
    "training_hours": 18
}
```
We can ask our model about this employee, like so:
```
response = requests.post('http://localhost:9696/predict', json=enrollee).json()

> {'target': False, 'target_probability': 0.20765447616577148}
```
Our model has told us that this employee only has about a 20% chance of staying with our company.

