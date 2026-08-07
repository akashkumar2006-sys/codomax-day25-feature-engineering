# codomax-day25-feature-engineering
# Codomax AI/ML Internship – Day 25

## Project
Feature Engineering for Machine Learning

## Description

This project demonstrates how raw data can be transformed into
more useful features before training a machine learning model.

A Random Forest classifier is trained using the engineered
features to predict student performance.

## Original Features

- Study Hours
- Attendance
- Assignments Completed
- Previous Score

## Engineered Features

### Academic Average
Combines previous score and assignment performance.

### Attendance Rate
Converts attendance percentage into a normalized rate.

### Study Efficiency
Calculates academic score relative to study hours.

## Concepts Covered

- Feature Engineering
- Feature Transformation
- Feature Creation
- Feature Scaling
- Random Forest
- Feature Importance
- Model Evaluation

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Learning Outcome

Learned how feature engineering can transform raw data into
more meaningful inputs for machine learning models.
