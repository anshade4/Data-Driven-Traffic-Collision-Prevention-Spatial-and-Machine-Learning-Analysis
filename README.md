# Data-Driven-Traffic-Collision-Prevention-Spatial-and-Machine-Learning-Analysis
## A Spatial and Machine Learning Approach to Hotspot Identification and Severity Prediction
It is a two-stage framework that turns raw police-reported collision records for Cambridgeshire, UK into decision-support outputs for road-safety teams. It shows where collision risk is concentrated and how severe a collision is likely to be, given its location and conditions.

## The Problem
Road-safety budgets are limited, so councils and police have to choose where to intervene and which conditions matter most. Most past studies treat hotspot detection and severity prediction as separate problems. This project links them: the hotspot risk found by unsupervised clustering is fed into the severity model as a predictive feature.

_Research question: How can spatial clustering and machine learning be combined to identify high-risk collision hotspots and predict the likely severity of collisions at those locations?_

## 1. Data Integration
Source: UK government open road-safety data (STATS19-derived) for Cambridgeshire, supplied as three linked sheets: Crashes (collision-level), Vehicles (vehicle-level) and Casualties (person-level).
Confirmed the Collision Reference Number was unique and used it as the primary key.
Pre-merge aggregation: a standard join would have multiplied rows because each collision has several vehicles and casualties. Instead, vehicle and casualty counts were aggregated per collision and then left-joined to the crash table. This kept all 9,870 collisions at one row per collision.
## 2. Data Cleaning & Validation
Cleaned column names. Converted date and time fields to datetime64, and Speed Limit and vehicle and casualty counts to numeric.
Grouped detailed categories (e.g. Weather, Lighting) into broader ones such as Daylight and Darkness: No Lighting to reduce sparsity.
Excluded fields with extreme missingness, e.g. Carriageway Hazards (>90% missing).
Checked value ranges (e.g. speed limits of 20–70 mph) and confirmed no duplicates after the merge.
## 3. Feature Engineering & Exploratory Data Analysis
Engineered features: Hour, Day_of_Week, Month, Year, Total_Vehicles, Total_Casualties and Risk_Zone.
What the data showed:
Severe class imbalance: about 7,000 Slight, 2,300 Serious and 200 Fatal collisions. This shaped the whole modelling strategy.
Speed: most collisions happen in 30 mph zones, but the share of fatal collisions rises with the speed limit.
Time: collisions peak in the rush hours (6–9 AM and 3–6 PM) and on Thursdays and Fridays. They drop sharply during the 2020 COVID-19 lockdowns and have not returned to pre-2020 levels.
Environment: most collisions happen in fine weather and daylight, but poor lighting and rain carry a higher share of serious outcomes.
Road type: single carriageways have the most collisions and a higher proportion of serious outcomes than roundabouts and dual carriageways.
## 4. Hotspot Identification (Unsupervised)
Reduced the encoded feature space to 3 dimensions with UMAP, which keeps local similarity between collisions.
Clustered with HDBSCAN. It needs no preset number of clusters, handles clusters of different densities and labels outliers as noise.
Turned the clusters into interpretable risk levels (low, medium or high) using a composite index of cluster size and severity-weighted counts (Fatal = 3, Serious = 2, Slight = 1).
Built a separate grid-based spatial risk layer from Easting/Northing coordinates, so the model gets an explicit geographic hotspot feature as well as the contextual clusters.
## 5. Severity Prediction (Supervised)
Accuracy is misleading here because a model that always predicts Slight scores well. The project therefore optimised macro-F1 and fatal-class recall instead.

Binary decomposition: separate detectors for Fatal and Serious, with Slight as the default class. This gives the rare classes dedicated modelling attention.
Models: XGBoost, LightGBM and CatBoost, using target encoding for categorical features.
Inputs: road, weather, lighting, time and vehicle features, plus the cluster-risk and spatial-risk features from Stage 4.
Optimisation: Optuna hyperparameter search, SMOTE resampling, bagging, threshold tuning and probability fusion across models.

The final model has an accuracy of 0.66, with class F1 scores of 0.778 (Slight), 0.345 (Serious) and 0.226 (Fatal). It correctly flagged 12 of the 47 fatal collisions in the held-out test set.

What didn't work: about ten other approaches were tested and rejected, including Balanced Random Forest, Isolation Forest, One-Class SVM, an Isolation Forest + XGBoost hybrid, BorderlineSMOTE and ADASYN with XGBoost, and a Gradient Boosting classifier. Joint Optuna tuning on its own actually lowered fatal recall. The gains came from decomposing the problem and fusing models, not from tuning harder.

## Limitations
Class imbalance: the test set has 1,446 Slight, 480 Serious and only 47 Fatal collisions. Fatal precision is still about 0.20, so the fatal-class gains are real but should be read with caution.
Missing exposure data: traffic flow, junction geometry and detailed road-environment variables were not available. These are the most likely route to further gains.
Single region: the results are specific to Cambridgeshire and have not been tested on other areas.
## Future Work
Add SHAP explainability to show which conditions drive fatal-risk predictions.
Bring in traffic-volume exposure data from DfT traffic counts.
Test whether hotspots persist over time and whether the model transfers to other UK regions.
