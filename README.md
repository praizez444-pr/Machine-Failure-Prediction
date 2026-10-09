#  Machine Failure Prediction & Predictive Maintenance System

An end-to-end machine learning project that uses machine operating conditions to predict whether a machine is likely to experience a failure.

The project uses the **AI4I 2020 Predictive Maintenance Dataset** and takes the process from data exploration and feature engineering to model comparison and deployment with **Streamlit**.

The main goal was not just to train a machine learning model, but to build something that could take real machine operating conditions and turn them into a simple, understandable prediction.

---
## Live Demo

Try the deployed application here: **[Machine Failure Prediction App](https://machine-failure-prediction-ocxx.onrender.com)**

Explore the application by entering machine operating data and viewing its predicted failure outcome.

##  Project Overview

Machine failures can lead to unexpected downtime, maintenance costs, and interruptions in production. Predictive maintenance provides a way to use historical machine data to identify conditions that may be associated with failure.

For this project, I built a classification pipeline that:

* Explores and cleans industrial machine data
* Engineers additional features from the available machine measurements
* Handles the imbalance between normal and failure records using **SMOTE**
* Trains and compares multiple classification algorithms
* Evaluates the models using metrics beyond accuracy
* Saves the selected model and preprocessing components
* Deploys the final model through an interactive **Streamlit web application**

The application allows a user to enter a machine's current operating conditions and receive a prediction of whether a failure is expected.

---

##  Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains **10,000 records** describing machine operating conditions and failure events.

### Dataset summary

| Description      |             Value |
| ---------------- | ----------------: |
| Total records    |            10,000 |
| Normal operation |             9,661 |
| Failure records  |               339 |
| Failure rate     |             3.39% |
| Target variable  | `Machine failure` |

The target variable is binary:

* `0` → No machine failure
* `1` → Machine failure

Because only 3.39% of the records represent failures, the dataset is highly imbalanced. This makes metrics such as **precision, recall, and F1-score** particularly important when evaluating the model.

### Data preprocessing

Before training the models, I:

* Removed `UDI` and `Product ID`, since they are identifiers rather than useful predictive features.
* Removed the failure mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, and `RNF`) to avoid giving the model information that directly indicates failure.
* Encoded the machine `Type` values (`L`, `M`, and `H`) using `LabelEncoder`.
* Created additional features based on the physical relationships between the machine measurements.

---

## 🔧 Feature Engineering

Instead of relying only on the original measurements, I created three additional features to capture more information about the machine's operating conditions.

### 1. Temperature Difference

The difference between process temperature and air temperature:

```text
Temperature Difference = Process Temperature - Air Temperature
```

This provides an indication of the thermal gradient under which the machine is operating.

### 2. Mechanical Power

Mechanical power is calculated from torque and rotational speed:

```text
Mechanical Power =
Torque × Rotational Speed × 2π / 60
```

This gives an estimate of the mechanical power being produced by the machine.

### 3. Torque × Tool Wear

This feature combines torque with accumulated tool wear:

```text
Torque Tool Wear = Torque × Tool Wear
```

The idea is to capture the relationship between the load being placed on the machine and the amount of wear accumulated by the tool.

These engineered features are also calculated automatically when a user enters values into the Streamlit application.

---

## Handling Class Imbalance

Only **339 out of 10,000 records** represent machine failures.

Training directly on this imbalanced dataset could cause a model to perform well on the majority class while doing a poor job of identifying actual failures.

To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data.

Importantly, SMOTE was applied to the training set rather than the test set, so that the final evaluation remained based on the original distribution of unseen data.

---

## Model Training & Evaluation

The dataset was divided into training and testing sets using an **80/20 stratified split** with `random_state=42`.

Five classification models were trained and compared:

* Random Forest
* Decision Tree
* Support Vector Machine (RBF)
* Logistic Regression
* Gaussian Naive Bayes

### Model comparison

| Model                |   Accuracy |  Precision |     Recall |   F1-Score |
| -------------------- | ---------: | ---------: | ---------: | ---------: |
| **Random Forest**    | **97.90%** | **64.44%** | **85.29%** | **73.42%** |
| Decision Tree        |     96.15% |     46.15% |     79.41% |     58.38% |
| SVM (RBF)            |     92.25% |     28.36% |     83.82% |     42.38% |
| Logistic Regression  |     86.10% |     17.99% |     86.76% |     29.80% |
| Gaussian Naive Bayes |     86.20% |     17.50% |     82.35% |     28.87% |

### Selected Model: Random Forest

Random Forest gave the best overall balance between identifying actual failures and avoiding unnecessary failure alarms.

Its performance on the held-out test set was:

* **Accuracy:** 97.90%
* **Precision:** 64.44%
* **Recall:** 85.29%
* **F1-score:** 73.42%

For predictive maintenance, recall is particularly important because missing an actual machine failure can be more costly than raising a false alarm.

The trained model, scaler, label encoder, and feature information were saved together using `joblib` in:

```text
machine_failure_model.pkl
```

---

## Streamlit Web Application

The trained model was turned into an interactive web application using **Streamlit**.

The application allows a user to enter machine operating conditions and receive a prediction without having to interact directly with the Python model code.

### Inputs

The application accepts:

| Input               | Range             |
| ------------------- | ----------------- |
| Machine Type        | `L`, `M`, `H`     |
| Air Temperature     | 295.3 – 304.5 K   |
| Process Temperature | 305.7 – 313.8 K   |
| Rotational Speed    | 1,168 – 2,886 rpm |
| Torque              | 3.8 – 76.6 Nm     |
| Tool Wear           | 0 – 253 min       |

These ranges are based on the values available in the dataset.

### Calculated indicators

After entering the operating conditions, the application automatically calculates:

* Temperature Difference
* Mechanical Power
* Torque × Tool Wear

These values are shown before the prediction so the user can see some of the operating indicators being used by the model.

The app then displays one of two outcomes:

```text
Machine failure predicted.
```

or

```text
No machine failure predicted for these operating conditions.
```

The application also provides information about the dataset and model performance in the sidebar.

---

## Project Structure

```text
machine-failure-prediction/
│
├── ai4i2020 (1).csv
│       # AI4I 2020 predictive maintenance dataset
│
├── machine_failure_model.pkl
│       # Saved model and preprocessing artifacts
│
├── app.py
│       # Streamlit web application
│
├── requirements.txt
│       # Project dependencies
│
└── README.md
        # Project documentation
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/machine-failure-prediction.git
cd machine-failure-prediction
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application should then open in your browser.

If it does not open automatically, Streamlit normally makes the application available at:

```text
http://localhost:8501
```

---

## Using the Application

Once the application is running:

1. Select the machine type.
2. Enter the air temperature.
3. Enter the process temperature.
4. Enter the rotational speed.
5. Enter the torque.
6. Enter the tool wear.
7. Review the calculated operating indicators.
8. Click **Predict machine failure**.
9. The application will display the predicted machine status.

The model uses the same feature transformations and preprocessing that were used during training, helping ensure that the inputs provided to the application are consistent with what the trained model expects.

---

## What I Learned From This Project

This project helped me move beyond simply training models in a notebook and think about the complete machine learning workflow.

Some of the key things I worked with were:

* Exploratory data analysis
* Data preprocessing
* Feature engineering
* Handling imbalanced datasets
* SMOTE
* Classification algorithms
* Model comparison
* Precision, recall, and F1-score
* Model serialization with `joblib`
* Building a user-facing ML application with Streamlit

One of the biggest lessons from the project was that **accuracy alone does not tell the whole story**, especially when the classes are highly imbalanced. A model can have high accuracy and still fail to identify the cases that matter most.

---

## Future Improvements

There are several ways this project could be improved further:

* Experiment with additional models and hyperparameter tuning.
* Add model explainability using techniques such as SHAP.
* Connect the system to live machine sensor data for continuous monitoring.

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Imbalanced-learn / SMOTE**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**

---

## Author

**Praise Abimbola**

This project is part of my journey into **Machine Learning and AI Engineering**, with an interest in applying machine learning to real-world engineering problems.

---

##  Project Status

**Completed with room for further improvement**

The current version includes the trained machine learning model, saved preprocessing components, and an interactive Streamlit application for machine failure prediction.
