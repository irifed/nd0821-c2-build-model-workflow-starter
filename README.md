# Build an ML Pipeline for Short-Term Rental Prices in NYC

Project 2 of the "Machine Learning DevOps Engineer" Udacity nanodegree.

Link to the Github repository of the project:
[https://github.com/irifed/nyc_airbnb](https://github.com/irifed/nyc_airbnb)

Link to the corresponding W&B project:
[https://wandb.ai/irifed/nyc_airbnb](https://wandb.ai/irifed/nyc_airbnb)


Steps to run the code:
```
mlflow run . -P steps=download
mlflow run . -P steps=basic_cleaning
```
In W&B, manually add a tag `reference` to the latest version of the clean_sample.csv artifact.
```
mlflow run . -P steps=data_check
mlflow run . -P steps=data_split
mlflow run . -P steps=train_random_forest
```

Testing the model on the test set: add `prod` to the latest trained model.
```
mlflow run . -P steps=test_regression_model
```