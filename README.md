# Predicting Intoxication from Motion Data

A machine learning project that predicts alcohol intoxication from smartphone accelerometer data to support **Just-in-Time Adaptive Interventions (JITAIs)**. Using motion-derived gait features, the project compares linear models, ensemble methods, and neural networks, showing that **tree-based models perform best**, reaching about **86.5% accuracy**.

Authors : Salma El Aazdoudi (Ecole Polytechnique-ENSAE) & Maria Micaela Linares (Ecole Polytechnique-ENSAE)

---

## Overview

Excessive alcohol consumption among college students remains a major public health concern, and many interventions fail because they do not happen at the moment risk emerges. This project explores whether **smartphone accelerometer data** can be used to detect intoxication in real time and support **preventive mobile interventions**.

The analysis uses motion and transdermal alcohol concentration (TAC) data collected from **13 college students** during a campus pub-crawl event. After extensive preprocessing and feature engineering, multiple machine learning models are benchmarked to evaluate how well intoxication can be inferred from gait and movement patterns alone.

---

## Research Question

**How can machine learning models leverage real-time smartphone sensor data to accurately predict alcohol intoxication risk, in order to trigger JITAIs only when intervention is likely to be useful?**

---

## Dataset

This project uses an open-source dataset introduced by **Killian, Passino, Nandi, Madden, and Clapp** in *Learning to Detect Heavy Drinking Episodes Using Smartphone Accelerometer Data*. The dataset combines physiological alcohol measurements with smartphone motion sensing collected during a real-world college drinking event.

It contains:

- **Transdermal alcohol concentration (TAC) readings** from ankle sensors, used to derive intoxication labels
- **Smartphone accelerometer data** capturing tri-axial motion patterns
- **Field-collected observations** recorded during a naturalistic campus pub-crawl setting

Following the original study, an observation is labeled as **intoxicated** when:

- **TAC > 0.08 g/dL**

---

## Preprocessing

The raw dataset was not immediately suitable for modeling and required substantial cleaning, alignment, and restructuring before analysis. Our preprocessing pipeline was partly inspired by the implementation available in the [detect_heavy_drinking](https://github.com/lisachua/detect_heavy_drinking) repository, while extending it to address irregular sampling, label propagation, and feature engineering more systematically.

### 1. Irregular sampling correction
Although the paper reports accelerometer data at **40 Hz**, timestamp inspection revealed highly irregular sampling, including long gaps. To address this, the motion signal was:

- resampled to **1000 Hz**
- imputed using **last-value carry-forward**

### 2. Class imbalance handling
Because TAC was sampled every **30 minutes**, sober states were overrepresented. To improve label coverage:

- TAC labels were propagated with **forward-fill imputation**
- intoxicated class representation increased from **24% to 36%**

### 3. Feature engineering
Using a two-level windowing approach:

- **10-second windows** were extracted
- each window was subdivided into **1-second segments**

Features were engineered in both domains:

#### Time-domain features
- mean
- median
- standard deviation
- min / max
- percentiles
- zero-crossing rate
- total energy
- energy entropy

#### Frequency-domain features
- spectral centroid
- spectral spread

Final dataset:

- **72,521 rows**
- **153 features**

---

## Models Evaluated

The project compares several model families:

### Linear models
- Logistic Regression
- Ridge Classifier
- Linear SVC
- Linear SVC with Random Fourier Features

### Ensemble methods
- Random Forest
- AdaBoost
- XGBoost
- Stacked Ensemble
- Voting Classifier

### Neural network
- Multilayer Perceptron (MLP)

---

## Results

Tree-based models clearly outperform linear models and the neural baseline.

| Model | Accuracy | Precision | Recall |
|------|---------:|----------:|-------:|
| Logistic Regression | 0.6344 | 0.4924 | 0.8403 |
| Ridge Classifier | 0.6342 | 0.4922 | 0.8432 |
| Linear SVC | 0.6347 | 0.4927 | 0.8429 |
| Linear SVC + Fourier Features | 0.6569 | 0.5109 | 0.8690 |
| Random Forest | 0.8650 | 0.8106 | 0.8104 |
| AdaBoost | 0.7408 | 0.5961 | 0.8462 |
| XGBoost | 0.8642 | 0.7956 | 0.8331 |
| MLP | 0.7754 | 0.7151 | 0.6144 |
| Stacked Ensemble | 0.8665 | 0.8200 | 0.8014 |
| Voting Classifier | 0.8665 | 0.8014 | 0.8318 |

### Best-performing models
- **Stacked Ensemble**: 86.65% accuracy
- **Voting Classifier**: 86.65% accuracy
- **Random Forest**: 86.50% accuracy
- **XGBoost**: 86.42% accuracy

---

## Key Findings

### 1. Rhythmic instability matters more than motion amplitude
The models rely less on how strongly subjects move and more on **how irregular or chaotic their motion becomes**. Frequency-domain features such as **spectral spread** are among the strongest predictors.

### 2. Loss of impact dampening is highly informative
Feature importance analysis suggests intoxication is associated with **undampened heel strikes** and more abrupt vertical deceleration patterns.

### 3. Tree-based models fit the data structure best
Because preprocessing created **piecewise-constant trajectories with abrupt transitions**, tree-based models were much better suited than linear models or MLPs.

### 4. Misclassifications reveal meaningful behavioral patterns
- **False positives** may reflect **early ataxia**, where movement already resembles intoxicated gait before TAC crosses the legal threshold
- **False negatives** may represent a **hyperactive instability phenotype**, where erratic high-energy motion is misread as sober activity

---

## Why This Matters for JITAIs

For JITAI systems, behavioral vulnerability may matter more than a strict chemical threshold. Even when the model misclassifies according to TAC labels, it may still detect **functional motor impairment**, which could be exactly when a real-time safety intervention is most useful.

This makes motion-based intoxication inference promising for:

- real-time behavioral risk detection
- mobile health interventions
- campus safety applications
- preventive notifications during high-risk events

---

## Limitations

This study has several important limitations:

- **Small sample size**: only 13 participants
- **No subject-specific normalization** for factors like height, weight, or BMI
- **No personalization** of sober baseline gait
- **Window independence assumption**, ignoring sequential intoxication progression
- **Context-specific dataset**, which may limit generalizability

Future work could explore:

- subject-wise normalization
- personalized baseline models
- sequential models such as **LSTMs**
- broader real-world validation

---

## Repository Structure

Example structure:

```bash
├── data/
├── notebooks/
├── src/
├── figures/
├── references/
├── README.md
└── report.pdf
