# Predictive-Maintenance-Project

The purpose of this project is to determine the best machine learning model we can use to predict the failure types of a mechanical system.

## Project outline:

1. Predictive Maintenance
2. System and Equipment.
3. Data Set Preview.
4. Exploratory Data Analysis (EDA).
5. Data Preprocessing.
6. Machine Learning Models.
7. Model Selection.

## Predictive Maintenace:

Data collected from product performance to detect anomalies and defects.
Machine learning models can be used on failure data to develop failure predictions.
This allows for:
- Low maintenance frequency.
- Prevention of unplanned reactive maintenance.
- Reduction in cost due to excessive preventative maintenance.

### Predictive Maintenance Data:

Predictive maintenance can use two types of data: 
- Visual: shows images of the equipment . Observable damage or wear are examples.
- Sensor: Devices in the measure key operations like Torque produced or Rotational Speed.

### Predictive Maintenance Challenges:

1. Acquiring the right data.
2. Understanding Machine Learning models.
3. Understanding equipment failure modes.
4. Predictive model integration.

## System and Data used in this project:

This project will utilize synthetic data that follows failure information encountered in the industry.
Assumptions: Motor system producing Torque and Rotational Speed.
Real life considerations:
- Ensuring correct measurements are used.
- System requirements.
- Understanding failure modes.

## Data Set Preview: 

We can use this data to build a machine learning model that can predict if the system under certain operating conditions would fail and what the failure type would be.

![image](https://user-images.githubusercontent.com/104313804/176327258-420397de-42d0-4793-9b11-47a39852a198.png)

![image](https://user-images.githubusercontent.com/104313804/176327318-e4cd9e23-297f-406e-8695-86fa3829dad9.png)

![image](https://user-images.githubusercontent.com/104313804/176327356-6c9ef0d3-4505-47b0-a6d9-8e415bcf9b1e.png)

### Undestanding the data preview and summary statistics:

Conclusions:
- No NaN or NULL values.
- The data types of each column fit well.
- 6 feature columns with determining values: Type, Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear.
- 2 target columns: Target, Failure Type.
- Data contains reasonable values.

### Data Set Preview:

Important data to consider: Type, Target and Failure Type Columns.
- Type: It contains 3 unique values that describe the part’s quality. 
  - H (High Quality, 50% of the data)
  - M (Medium Quality, 30% of the data).
  - L (Low Quality,  20% of the data).
- Target: Determines if the part failed under the specified conditions. 
  - 0 (No failure).
  - 1 (Failure).
- Failure Type: Determines the type of failure that occurred. 
  - No Failure.
  - Power Failure.
  - Tool Wear Failure.
  - Overstrain Failure.
  - Random Failures.
  - Heat Dissipation Failure.

## Exploratory Data Analysis:

We delete the UDI and ID columns since they do not provide any predictive value to our analysis.
We also have 2 target columns (Target and Failure Type) this would generate memory leak.
Decided to use the Failure Type column as our target column.

![image](https://user-images.githubusercontent.com/104313804/176328313-3b543329-ad0b-4287-bf48-12b8de9007a4.png)

### Heat Map: Correlation Coefficients.

![image](https://user-images.githubusercontent.com/104313804/176328575-5330726a-1415-480a-a271-cebfe2f04705.png)

### Data Features Visualization:

![image](https://user-images.githubusercontent.com/104313804/176328830-c18d2962-9b55-4371-a641-ff2f14b5f14e.png)

Observations: 
- Failure occurs at extreme measurements.
- Air temperature and process temperature have positive correlation.
- Torque and rotational speed have a heavy negative correlation.

Cautions: 
- Scale difference in measurements results in bias.

### Features Count Plots
![image](https://user-images.githubusercontent.com/104313804/176329269-cba0e493-6b3c-412c-88b0-61a8f9f43c79.png)


Observation:
- Failure type “No Failure” count is  a lot higher than any other failure types.
- This shows the data is imbalanced. 
- This is a normal scenario in real life failure data.

Failure Type unique values:

![image](https://user-images.githubusercontent.com/104313804/176329184-60dc9e55-5ac8-47fb-92a6-0c9d2740e522.png)

## Data Preprocessing:

Encoding the Part Type and Failure Type: In order to apply machine learning models, we need to make sure to encode categorical data to ensure they are represented by integer values.

Using pd.get_dummies():

- L : Type_L = 1, Type_M=0.
- M : Type_L = 0, Type_M=1.
- H : Type_L = 0, Type_M=0.

Using LabelEncoder(): 
- 0 : Heat Dissipation Failure
- 1 : No Failure
- 2 : Overstrain Failure
- 3 : Power Failure
- 4 : Random Failures
- 5 : Tool Wear Failure

![image](https://user-images.githubusercontent.com/104313804/176329484-ad7d2ed1-db28-430a-b48e-5167ffd69db4.png)

### Dealing with unbalanced data:

Oversampling was implemented using SMOTETomek from imblearn.combine python package.
This allows us to create synthetic data and balance our data set. 
It uses K-nearest neighbors algorithm from the SMOTE technique, and removes unnecessary outliers with the Tomek links.

Below we can observe the data before balancing and after balancing was implemented:
![image](https://user-images.githubusercontent.com/104313804/178612976-62dd56fb-deb2-4235-8a66-6ce3c4928fc0.png)

### Training and Test Values:

Using the train_test_split function from the sklearn.model_selection package we can separate the data into 70% training data and 30% test data.
We will use the balanced training set to train the model (X_train_res, y_train_res) and the normal set in our test set (X_test, y_test)

### Checking for Outliers:
In order to verify the presence of outliers in the data, we plot the distribution of each data feature for each failure type. 

![image](https://user-images.githubusercontent.com/104313804/178613157-0291891b-5f5e-4544-8ddd-41fa5127f25f.png)
![image](https://user-images.githubusercontent.com/104313804/178613169-7f6f48ce-e2c3-4dd8-81ab-48c883e8c139.png)
![image](https://user-images.githubusercontent.com/104313804/178613186-ae84b6d6-7f50-48ff-8fda-cf00ce1e285e.png)
![image](https://user-images.githubusercontent.com/104313804/178613205-3820b44a-6664-472f-ab01-38870d8466fb.png)
![image](https://user-images.githubusercontent.com/104313804/178613218-92a50bd7-a0e1-4358-aec1-98c707a89051.png)

As seen above, there is a strong presence of outliers in the data, which means that we will need to use scaling techniques that take care of this for us.
For this project I will be using RobustScaling when scaling the data.

## Machine Learning Models: 

In this project I will use 3 machine learning non-linear classifiers and determine which produce the best results for our predictive maintenance scenario. We will utilize:
- K-Nearest Neighbors (KNN) Classifier.
- Decision Tree Classifier.
- XGBoost Classifier.
It is important to note that there are a lot of different models that we could try for the data and in order to hone the scope of this project I will be focusing on these three types.

### Machine Learning Model Application: 

In order to train each machine learning model we used the balanced training set. This is done in order to ensure there is enough data to train the data set in all the failure types shown in the failure data.
In order to test each machine learning model we used the unbalanced test set. This is to ensure that the models are being tested with representative data to real life scenario. In real life data collection, the trend of unbalanced data will be present.

- Training Data: X_train_res, y_train_res.
- Test Data: X_test, y_test.
Using scaled and unscaled data, obtained: 
- Accuracy Score.
- 5-fold Cross Validation.
- Classification report

### Observations: 

![image](https://user-images.githubusercontent.com/104313804/178614266-36ca432c-f497-41b8-b5b5-60f6ec0d13ce.png)
![image](https://user-images.githubusercontent.com/104313804/178614277-2816dcc4-fa86-4916-a2e9-e4abd5196e45.png)

As shown in the graphs abov:
- When using KNN Classifier, using scaled data to train the model, proves to have a greater accuracy score.
- Accuracy is the same for unscaled and scaled data when using the Decision Tree and the XGBoost Classifiers, this is due to the fact unscaled data does not affec the performance of these more powerful models.
- 5-fold Cross-validation show to develop high accuracy for all models with very neglegible difference in score, proving the advantages of using k-fold cross-validation to create more accurate models.

# Model Selection:

As shown below, XGBoost model performed the best between all three models under the default parameter conditions of each model.
Since we are dealing with unbalanced data, we need to understand that the accuracy score is not the best parameter to select the model, since it is skewed towards the higher quantity data, in this case the "No Failure" failure type. Because of this I compared the precision of each model for each failure type, and the model and conditions with the best overall score was selected as the best for a predictive maintenance system, in this case XGBoost.

Future Considerations: 
- Parameter Tuning.
- Using GridSearchCV.

![image](https://user-images.githubusercontent.com/104313804/178614908-fa5b4bed-a95a-408e-bf8d-28b259159136.png)
![image](https://user-images.githubusercontent.com/104313804/178614917-b409fc6e-5be4-492d-8d0a-caec4a931406.png)
![image](https://user-images.githubusercontent.com/104313804/178614921-beaca5a5-c758-42ed-a88c-928cfc165ca4.png)












