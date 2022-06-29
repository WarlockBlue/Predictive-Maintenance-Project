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
0 : Heat Dissipation Failure
1 : No Failure
2 : Overstrain Failure
3 : Power Failure
4 : Random Failures
5 : Tool Wear Failure

![image](https://user-images.githubusercontent.com/104313804/176329484-ad7d2ed1-db28-430a-b48e-5167ffd69db4.png)



