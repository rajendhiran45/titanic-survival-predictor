# Titanic Survival Predictor

An interactive **Machine Learning web application** that predicts whether a Titanic passenger would survive based on passenger details such as age, gender, class, fare, and family information.

This project demonstrates the complete **data science workflow** including data preprocessing, feature engineering, model training, and deployment using **Streamlit**.

---

## Project Overview

The Titanic dataset is one of the most well-known datasets used in machine learning.
This project builds a **Random Forest Classification model** to predict passenger survival using historical data.

The application allows users to input passenger details through a **Streamlit interface** and instantly receive a survival prediction.

---

## Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **Scikit-learn** – Machine learning model
* **Matplotlib** – Data visualization
* **Streamlit** – Web application interface

---

## Features

* Data cleaning and preprocessing
* Feature engineering (FamilySize, IsAlone)
* Random Forest machine learning model
* Feature importance visualization
* Interactive web interface with Streamlit
* Real-time survival prediction

---

## Project Structure

```
titanic-survival-predictor
│
├── titanic_model
│   ├── app.py                # Streamlit web application
│   ├── titanic.py            # Data processing & ML model
│   └── Titanic-Dataset.csv   # Dataset
│
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1.Clone the repository

```
git clone https://github.com/rajendhiran45/titanic-survival-predictor.git
```

### 2.Navigate to the project folder

```
cd titanic-survival-predictor/titanic_model
```

### 3️.Install dependencies

```
pip install -r requirements.txt
```

### 4️.Run the Streamlit app

```
streamlit run app.py
```

The application will open in your browser.

---

## Machine Learning Model

The model used in this project:

**Random Forest Classifier**

Key features used for prediction:

* Passenger Class
* Sex
* Age
* Fare
* Number of Siblings/Spouse
* Number of Parents/Children
* Family Size
* Is Alone
* Embarked Port

---

## Example Prediction

Users can input passenger information such as:

* Passenger name
* Age
* Passenger class
* Gender
* Fare
* Family members aboard
* Embarkation port

The application then predicts whether the passenger is **likely to survive or not**.

---

## Future Improvements

* Display survival probability percentage
* Add interactive charts and dashboards
* Deploy the application online
* Improve model accuracy with additional features

---

## Author

**Rajendhiran**

Electronics and Communication Engineering student interested in **AI, Machine Learning, and Software Development**.

GitHub:
https://github.com/rajendhiran45

Live Demo:https://titanic-survival-predictor-ai.streamlit.app/
---

⭐ If you like this project, consider giving it a **star on GitHub**!
