***
# Supervised Learning : Leveraging Ensemble Learning With Bagging, Boosting, Stacking and Blending Approaches

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *March 8, 2025*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Splitting](#1.4.1)
        * [1.4.2 Data Profiling](#1.4.2)
        * [1.4.3 Category Aggregation and Encoding](#1.4.3)
        * [1.4.4 Outlier and Distributional Shape Analysis](#1.4.4)
        * [1.4.5 Collinearity](#1.4.5)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Premodelling Data Preparation](#1.6)
        * [1.6.1 Preprocessed Data Description](#1.6.1)
        * [1.6.2 Preprocessing Pipeline Development](#1.6.2)
    * [1.7 Bagged Model Development](#1.7)
        * [1.7.1 Random Forest](#1.7.1)
        * [1.7.2 Extra Trees](#1.7.2)
        * [1.7.3 Bagged Decision Tree](#1.7.3)
        * [1.7.4 Bagged Logistic Regression](#1.7.4)
        * [1.7.5 Bagged Support Vector Machine](#1.7.5)
    * [1.8 Boosted Model Development](#1.8)
        * [1.8.1 AdaBoost](#1.8.1)
        * [1.8.2 Gradient Boosting](#1.8.2)
        * [1.8.3 XGBoost](#1.8.3)
        * [1.8.4 Light GBM](#1.8.4)
        * [1.8.5 CatBoost](#1.8.5)
    * [1.9 Stacked Model Development](#1.9)
        * [1.9.1 Base Learner - K-Nearest Neighbors](#1.9.1)
        * [1.9.2 Base Learner - Support Vector Machine](#1.9.2)
        * [1.9.3 Base Learner - Ridge Classifier](#1.9.3)
        * [1.9.4 Base Learner - Neural Network](#1.9.4)
        * [1.9.5 Base Learner - Decision Tree](#1.9.5)
        * [1.9.6 Meta Learner - Logistic Regression](#1.9.6)
    * [1.10 Blended Model Development](#1.10)
        * [1.10.1 Base Learner - K-Nearest Neighbors](#1.10.1)
        * [1.10.2 Base Learner - Support Vector Machine](#1.10.2)
        * [1.10.3 Base Learner - Ridge Classifier](#1.10.3)
        * [1.10.4 Base Learner - Neural Network](#1.10.4)
        * [1.10.5 Base Learner - Decision Tree](#1.10.5)
        * [1.10.6 Meta Learner - Logistic Regression](#1.10.6)
    * [1.11 Consolidated Findings](#1.11)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***


# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores different **Ensemble Learning** approaches which combine the predictions from multiple models in an effort to achieve better predictive performance using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark>. The ensemble frameworks applied in the analysis were grouped into three classes including the **Bagging Approach** which fits many individual learners on different samples of the same dataset and averages the predictions; **Boosting Approach** which adds ensemble members sequentially that correct the predictions made by prior models and outputs a weighted average of the predictions;  and **Stacking or Blending Approach** which consolidates many different and diverse learners on the same data and uses another model to learn how to best combine the predictions. Bagged models applied were the **Random Forest**, **Extra Trees**, **Bagged Decision Tree**, **Bagged Logistic Regression** and **Bagged Support Vector Machine** algorithms. Boosting models included the **AdaBoost**, **Stochastic Gradient Boosting**, **Extreme Gradient Boosting**, **Light Gradient Boosting Machines** and **CatBoost** algorithms.  Individual base learners including the **K-Nearest Neighbors**, **Support Vector Machine**, **Ridge Classifier**, **Neural Network** and **Decision Tree** algorithms were stacked or blended together as contributors to the **Logistic Regression** meta-model. The resulting predictions derived from all ensemble learning models were independtly evaluated on a test set based on accuracy and F1 score metrics. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[Ensemble Learning](https://www.manning.com/books/ensemble-methods-for-machine-learning) is a machine learning technique that improves predictive accuracy by combining multiple models to leverage their collective strengths. Traditional machine learning models often struggle with either high bias, which leads to overly simplistic predictions, or high variance, which makes them too sensitive to fluctuations in the data. Ensemble learning addresses these challenges by aggregating the outputs of several models, creating a more robust and reliable predictor. In classification problems, this can be done through majority voting, weighted averaging, or more advanced meta-learning techniques. The key advantage of ensemble learning is its ability to reduce both bias and variance, leading to better generalization on unseen data. However, this comes at the cost of increased computational complexity and interpretability, as managing multiple models requires more resources and makes it harder to explain predictions.

[Bagging](https://www.manning.com/books/ensemble-methods-for-machine-learning), or Bootstrap Aggregating, is an ensemble learning technique that reduces model variance by training multiple instances of the same algorithm on different randomly sampled subsets of the training data. The fundamental problem bagging aims to solve is overfitting, particularly in high-variance models. By generating multiple bootstrap samples—random subsets created through sampling with replacement — bagging ensures that each model is trained on slightly different data, making the overall prediction more stable. In classification problems, the final output is obtained by majority voting among the individual models, while in regression, their predictions are averaged. Bagging is particularly effective when dealing with noisy datasets, as it smooths out individual model errors. However, its effectiveness is limited for low-variance models, and the requirement to train multiple models increases computational cost.

[Boosting](https://www.manning.com/books/ensemble-methods-for-machine-learning) is an ensemble learning method that builds a strong classifier by training models sequentially, where each new model focuses on correcting the mistakes of its predecessors. Boosting assigns higher weights to misclassified instances, ensuring that subsequent models pay more attention to these hard-to-classify cases. The motivation behind boosting is to reduce both bias and variance by iteratively refining weak learners — models that perform only slightly better than random guessing — until they collectively form a strong classifier. In classification tasks, predictions are refined by combining weighted outputs of multiple weak models, typically decision stumps or shallow trees. This makes boosting highly effective in uncovering complex patterns in data. However, the sequential nature of boosting makes it computationally expensive compared to bagging, and it is more prone to overfitting if the number of weak learners is too high.

[Stacking](https://www.manning.com/books/ensemble-methods-for-machine-learning), or stacked generalization, is an advanced ensemble method that improves predictive performance by training a meta-model to learn the optimal way to combine multiple base models using their out-of-fold predictions. Unlike traditional ensemble techniques such as bagging and boosting, which aggregate predictions through simple rules like averaging or majority voting, stacking introduces a second-level model that intelligently learns how to integrate diverse base models. The process starts by training multiple classifiers on the training dataset. However, instead of directly using their predictions, stacking employs k-fold cross-validation to generate out-of-fold predictions. Specifically, each base model is trained on a subset of the training data while leaving out a validation fold, and predictions on that unseen fold are recorded. This process is repeated across all folds, ensuring that each instance in the training data receives predictions from models that never saw it during training. These out-of-fold predictions are then used as input features for a meta-model, which learns the best way to combine them into a final decision. The advantage of stacking is that it allows different models to complement each other, capturing diverse aspects of the data that a single model might miss. This often results in superior classification accuracy compared to individual models or simpler ensemble approaches. However, stacking is computationally expensive, requiring multiple training iterations for base models and the additional meta-model. It also demands careful tuning to prevent overfitting, as the meta-model’s complexity can introduce new sources of error. Despite these challenges, stacking remains a powerful technique in applications where maximizing predictive performance is a priority.

[Blending](https://www.manning.com/books/ensemble-methods-for-machine-learning) is an ensemble technique that enhances classification accuracy by training a meta-model on a holdout validation set, rather than using out-of-fold predictions like stacking. This simplifies implementation while maintaining the benefits of combining multiple base models. The process of blending starts by training base models on the full training dataset. Instead of applying cross-validation to obtain out-of-fold predictions, blending reserves a small portion of the training data as a holdout set. The base models make predictions on this unseen holdout set, and these predictions are then used as input features for a meta-model, which learns how to optimally combine them into a final classification decision. Since the meta-model is trained on predictions from unseen data, it avoids the risk of overfitting that can sometimes occur when base models are evaluated on the same data they were trained on. Blending is motivated by its simplicity and ease of implementation compared to stacking, as it eliminates the need for repeated k-fold cross-validation to generate training data for the meta-model. However, one drawback is that the meta-model has access to fewer training examples, as a portion of the data is withheld for validation rather than being used for training. This can limit the generalization ability of the final model, especially if the holdout set is too small. Despite this limitation, blending remains a useful approach in applications where a quick and effective ensemble method is needed without the computational overhead of stacking.


## 1.1. Data Background <a class="anchor" id="1.1"></a>

An open [Thyroid Disease Dataset](https://www.kaggle.com/datasets/jainaru/thyroid-disease-data/data) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Jai Naru](https://www.kaggle.com/jainaru) and [Abuchi Onwuegbusi](https://www.kaggle.com/datasets/abuchionwuegbusi/thyroid-cancer-recurrence-prediction/data)) was used for the analysis as consolidated from the following primary sources: 
1. Reference Repository entitled **Differentiated Thyroid Cancer Recurrence** from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)
2. Research Paper entitled **Machine Learning for Risk Stratification of Thyroid Cancer Patients: a 15-year Cohort Study** from the [European Archives of Oto-Rhino-Laryngology](https://link.springer.com/article/10.1007/s00405-023-08299-w)

This study hypothesized that the various clinicopathological characteristics influence differentiated thyroid cancer recurrence between patients.

The dichotomous categorical variable for the study is:
* <span style="color: #FF0000">Recurred</span> - Status of the patient (Yes, Recurrence of differentiated thyroid cancer | No, No recurrence of differentiated thyroid cancer)

The predictor variables for the study are:
* <span style="color: #FF0000">Age</span> - Patient's age (Years)
* <span style="color: #FF0000">Gender</span> - Patient's sex (M | F)
* <span style="color: #FF0000">Smoking</span> - Indication of smoking (Yes | No)
* <span style="color: #FF0000">Hx Smoking</span> - Indication of smoking history (Yes | No)
* <span style="color: #FF0000">Hx Radiotherapy</span> - Indication of radiotherapy history for any condition (Yes | No)
* <span style="color: #FF0000">Thyroid Function</span> - Status of thyroid function (Clinical Hyperthyroidism, Hypothyroidism | Subclinical Hyperthyroidism, Hypothyroidism | Euthyroid)
* <span style="color: #FF0000">Physical Examination</span> - Findings from physical examination including palpation of the thyroid gland and surrounding structures (Normal | Diffuse Goiter | Multinodular Goiter | Single Nodular Goiter Left, Right)
* <span style="color: #FF0000">Adenopathy</span> - Indication of enlarged lymph nodes in the neck region (No | Right | Extensive | Left | Bilateral | Posterior)
* <span style="color: #FF0000">Pathology</span> - Specific thyroid cancer type as determined by pathology examination of biopsy samples (Follicular | Hurthel Cell | Micropapillary | Papillary)
* <span style="color: #FF0000">Focality</span> - Indication if the cancer is limited to one location or present in multiple locations (Uni-Focal | Multi-Focal)
* <span style="color: #FF0000">Risk</span> - Risk category of the cancer based on various factors, such as tumor size, extent of spread, and histological type (Low | Intermediate | High)
* <span style="color: #FF0000">T</span> - Tumor classification based on its size and extent of invasion into nearby structures (T1a | T1b | T2 | T3a | T3b | T4a | T4b)
* <span style="color: #FF0000">N</span> - Nodal classification indicating the involvement of lymph nodes (N0 | N1a | N1b)
* <span style="color: #FF0000">M</span> - Metastasis classification indicating the presence or absence of distant metastases (M0 | M1)
* <span style="color: #FF0000">Stage</span> - Overall stage of the cancer, typically determined by combining T, N, and M classifications (I | II | III | IVa | IVb)
* <span style="color: #FF0000">Response</span> - Cancer's response to treatment (Biochemical Incomplete | Indeterminate | Excellent | Structural Incomplete)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The initial tabular dataset was comprised of 383 observations and 17 variables (including 1 target and 16 predictors).
    * **383 rows** (observations)
    * **17 columns** (variables)
        * **1/17 target** (categorical)
             * <span style="color: #FF0000">Recurred</span>
        * **1/17 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
        * **16/17 predictor** (categorical)
             * <span style="color: #FF0000">Gender</span>
             * <span style="color: #FF0000">Smoking</span>
             * <span style="color: #FF0000">Hx_Smoking</span>
             * <span style="color: #FF0000">Hx_Radiotherapy</span>
             * <span style="color: #FF0000">Thyroid_Function</span>
             * <span style="color: #FF0000">Physical_Examination</span>
             * <span style="color: #FF0000">Adenopathy</span>
             * <span style="color: #FF0000">Pathology</span>
             * <span style="color: #FF0000">Focality</span>
             * <span style="color: #FF0000">Risk</span>
             * <span style="color: #FF0000">T</span>
             * <span style="color: #FF0000">N</span>
             * <span style="color: #FF0000">M</span>
             * <span style="color: #FF0000">Stage</span>
             * <span style="color: #FF0000">Response</span>
            																


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import itertools
import os
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats
from scipy.stats import pointbiserialr, chi2_contingency

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score

```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\original"
DATASETS_FINAL_PATH = r"datasets\final\complete"
DATASETS_FINAL_TRAIN_PATH = r"datasets\final\train"
DATASETS_FINAL_TRAIN_FEATURES_PATH = r"datasets\final\train\features"
DATASETS_FINAL_TRAIN_TARGET_PATH = r"datasets\final\train\target"
DATASETS_FINAL_VALIDATION_PATH = r"datasets\final\validation"
DATASETS_FINAL_VALIDATION_FEATURES_PATH = r"datasets\final\validation\features"
DATASETS_FINAL_VALIDATION_TARGET_PATH = r"datasets\final\validation\target"
DATASETS_FINAL_TEST_PATH = r"datasets\final\test"
DATASETS_FINAL_TEST_FEATURES_PATH = r"datasets\final\test\features"
DATASETS_FINAL_TEST_TARGET_PATH = r"datasets\final\test\target"
DATASETS_PREPROCESSED_PATH = r"datasets\preprocessed"
DATASETS_PREPROCESSED_TRAIN_PATH = r"datasets\preprocessed\train"
DATASETS_PREPROCESSED_TRAIN_FEATURES_PATH = r"datasets\preprocessed\train\features"
DATASETS_PREPROCESSED_TRAIN_TARGET_PATH = r"datasets\preprocessed\train\target"
DATASETS_PREPROCESSED_VALIDATION_PATH = r"datasets\preprocessed\validation"
DATASETS_PREPROCESSED_VALIDATION_FEATURES_PATH = r"datasets\preprocessed\validation\features"
DATASETS_PREPROCESSED_VALIDATION_TARGET_PATH = r"datasets\preprocessed\validation\target"
DATASETS_PREPROCESSED_TEST_PATH = r"datasets\preprocessed\test"
DATASETS_PREPROCESSED_TEST_FEATURES_PATH = r"datasets\preprocessed\test\features"
DATASETS_PREPROCESSED_TEST_TARGET_PATH = r"datasets\preprocessed\test\target"
MODELS_PATH = r"models"

```


```python
##################################
# Loading the dataset
# from the DATASETS_ORIGINAL_PATH
##################################
thyroid_cancer = pd.read_csv(os.path.join("..", DATASETS_ORIGINAL_PATH, "Thyroid_Diff.csv"))

```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(thyroid_cancer.shape)

```

    Dataset Dimensions: 
    


    (383, 17)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(thyroid_cancer.dtypes)

```

    Column Names and Data Types:
    


    Age                      int64
    Gender                  object
    Smoking                 object
    Hx Smoking              object
    Hx Radiotherapy         object
    Thyroid Function        object
    Physical Examination    object
    Adenopathy              object
    Pathology               object
    Focality                object
    Risk                    object
    T                       object
    N                       object
    M                       object
    Stage                   object
    Response                object
    Recurred                object
    dtype: object



```python
##################################
# Renaming and standardizing the column names
# to replace blanks with undercores
##################################
thyroid_cancer.columns = thyroid_cancer.columns.str.replace(" ", "_")

```


```python
##################################
# Taking a snapshot of the dataset
##################################
thyroid_cancer.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Indeterminate</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Selecting categorical columns (both object and categorical types)
# and listing the unique categorical levels
##################################
cat_cols = thyroid_cancer.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    print(f"Categorical | Object Column: {col}")
    print(thyroid_cancer[col].unique())  
    print("-" * 40)
    
```

    Categorical | Object Column: Gender
    ['F' 'M']
    ----------------------------------------
    Categorical | Object Column: Smoking
    ['No' 'Yes']
    ----------------------------------------
    Categorical | Object Column: Hx_Smoking
    ['No' 'Yes']
    ----------------------------------------
    Categorical | Object Column: Hx_Radiotherapy
    ['No' 'Yes']
    ----------------------------------------
    Categorical | Object Column: Thyroid_Function
    ['Euthyroid' 'Clinical Hyperthyroidism' 'Clinical Hypothyroidism'
     'Subclinical Hyperthyroidism' 'Subclinical Hypothyroidism']
    ----------------------------------------
    Categorical | Object Column: Physical_Examination
    ['Single nodular goiter-left' 'Multinodular goiter'
     'Single nodular goiter-right' 'Normal' 'Diffuse goiter']
    ----------------------------------------
    Categorical | Object Column: Adenopathy
    ['No' 'Right' 'Extensive' 'Left' 'Bilateral' 'Posterior']
    ----------------------------------------
    Categorical | Object Column: Pathology
    ['Micropapillary' 'Papillary' 'Follicular' 'Hurthel cell']
    ----------------------------------------
    Categorical | Object Column: Focality
    ['Uni-Focal' 'Multi-Focal']
    ----------------------------------------
    Categorical | Object Column: Risk
    ['Low' 'Intermediate' 'High']
    ----------------------------------------
    Categorical | Object Column: T
    ['T1a' 'T1b' 'T2' 'T3a' 'T3b' 'T4a' 'T4b']
    ----------------------------------------
    Categorical | Object Column: N
    ['N0' 'N1b' 'N1a']
    ----------------------------------------
    Categorical | Object Column: M
    ['M0' 'M1']
    ----------------------------------------
    Categorical | Object Column: Stage
    ['I' 'II' 'IVB' 'III' 'IVA']
    ----------------------------------------
    Categorical | Object Column: Response
    ['Indeterminate' 'Excellent' 'Structural Incomplete'
     'Biochemical Incomplete']
    ----------------------------------------
    Categorical | Object Column: Recurred
    ['No' 'Yes']
    ----------------------------------------
    


```python
##################################
# Correcting a category level
##################################
thyroid_cancer["Pathology"] = thyroid_cancer["Pathology"].replace("Hurthel cell", "Hurthle Cell")

```


```python
##################################
# Setting the levels of the categorical variables
##################################
thyroid_cancer['Recurred'] = thyroid_cancer['Recurred'].astype('category')
thyroid_cancer['Recurred'] = thyroid_cancer['Recurred'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Gender'] = thyroid_cancer['Gender'].astype('category')
thyroid_cancer['Gender'] = thyroid_cancer['Gender'].cat.set_categories(['M', 'F'], ordered=True)
thyroid_cancer['Smoking'] = thyroid_cancer['Smoking'].astype('category')
thyroid_cancer['Smoking'] = thyroid_cancer['Smoking'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Hx_Smoking'] = thyroid_cancer['Hx_Smoking'].astype('category')
thyroid_cancer['Hx_Smoking'] = thyroid_cancer['Hx_Smoking'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Hx_Radiotherapy'] = thyroid_cancer['Hx_Radiotherapy'].astype('category')
thyroid_cancer['Hx_Radiotherapy'] = thyroid_cancer['Hx_Radiotherapy'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Thyroid_Function'] = thyroid_cancer['Thyroid_Function'].astype('category')
thyroid_cancer['Thyroid_Function'] = thyroid_cancer['Thyroid_Function'].cat.set_categories(['Euthyroid', 'Subclinical Hypothyroidism', 'Subclinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Clinical Hyperthyroidism'], ordered=True)
thyroid_cancer['Physical_Examination'] = thyroid_cancer['Physical_Examination'].astype('category')
thyroid_cancer['Physical_Examination'] = thyroid_cancer['Physical_Examination'].cat.set_categories(['Normal', 'Single nodular goiter-left', 'Single nodular goiter-right', 'Multinodular goiter', 'Diffuse goiter'], ordered=True)
thyroid_cancer['Adenopathy'] = thyroid_cancer['Adenopathy'].astype('category')
thyroid_cancer['Adenopathy'] = thyroid_cancer['Adenopathy'].cat.set_categories(['No', 'Left', 'Right', 'Bilateral', 'Posterior', 'Extensive'], ordered=True)
thyroid_cancer['Pathology'] = thyroid_cancer['Pathology'].astype('category')
thyroid_cancer['Pathology'] = thyroid_cancer['Pathology'].cat.set_categories(['Hurthle Cell', 'Follicular', 'Micropapillary', 'Papillary'], ordered=True)
thyroid_cancer['Focality'] = thyroid_cancer['Focality'].astype('category')
thyroid_cancer['Focality'] = thyroid_cancer['Focality'].cat.set_categories(['Uni-Focal', 'Multi-Focal'], ordered=True)
thyroid_cancer['Risk'] = thyroid_cancer['Risk'].astype('category')
thyroid_cancer['Risk'] = thyroid_cancer['Risk'].cat.set_categories(['Low', 'Intermediate', 'High'], ordered=True)
thyroid_cancer['T'] = thyroid_cancer['T'].astype('category')
thyroid_cancer['T'] = thyroid_cancer['T'].cat.set_categories(['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'], ordered=True)
thyroid_cancer['N'] = thyroid_cancer['N'].astype('category')
thyroid_cancer['N'] = thyroid_cancer['N'].cat.set_categories(['N0', 'N1a', 'N1b'], ordered=True)
thyroid_cancer['M'] = thyroid_cancer['M'].astype('category')
thyroid_cancer['M'] = thyroid_cancer['M'].cat.set_categories(['M0', 'M1'], ordered=True)
thyroid_cancer['Stage'] = thyroid_cancer['Stage'].astype('category')
thyroid_cancer['Stage'] = thyroid_cancer['Stage'].cat.set_categories(['I', 'II', 'III', 'IVA', 'IVB'], ordered=True)
thyroid_cancer['Response'] = thyroid_cancer['Response'].astype('category')
thyroid_cancer['Response'] = thyroid_cancer['Response'].cat.set_categories(['Excellent', 'Structural Incomplete', 'Biochemical Incomplete', 'Indeterminate'], ordered=True)

```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(thyroid_cancer.describe(include='number').transpose())

```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>383.0</td>
      <td>40.866841</td>
      <td>15.134494</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>37.0</td>
      <td>51.0</td>
      <td>82.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variables
##################################
print('Categorical Variable Summary:')
display(thyroid_cancer.describe(include='category').transpose())

```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gender</th>
      <td>383</td>
      <td>2</td>
      <td>F</td>
      <td>312</td>
    </tr>
    <tr>
      <th>Smoking</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>334</td>
    </tr>
    <tr>
      <th>Hx_Smoking</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>355</td>
    </tr>
    <tr>
      <th>Hx_Radiotherapy</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>376</td>
    </tr>
    <tr>
      <th>Thyroid_Function</th>
      <td>383</td>
      <td>5</td>
      <td>Euthyroid</td>
      <td>332</td>
    </tr>
    <tr>
      <th>Physical_Examination</th>
      <td>383</td>
      <td>5</td>
      <td>Single nodular goiter-right</td>
      <td>140</td>
    </tr>
    <tr>
      <th>Adenopathy</th>
      <td>383</td>
      <td>6</td>
      <td>No</td>
      <td>277</td>
    </tr>
    <tr>
      <th>Pathology</th>
      <td>383</td>
      <td>4</td>
      <td>Papillary</td>
      <td>287</td>
    </tr>
    <tr>
      <th>Focality</th>
      <td>383</td>
      <td>2</td>
      <td>Uni-Focal</td>
      <td>247</td>
    </tr>
    <tr>
      <th>Risk</th>
      <td>383</td>
      <td>3</td>
      <td>Low</td>
      <td>249</td>
    </tr>
    <tr>
      <th>T</th>
      <td>383</td>
      <td>7</td>
      <td>T2</td>
      <td>151</td>
    </tr>
    <tr>
      <th>N</th>
      <td>383</td>
      <td>3</td>
      <td>N0</td>
      <td>268</td>
    </tr>
    <tr>
      <th>M</th>
      <td>383</td>
      <td>2</td>
      <td>M0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>Stage</th>
      <td>383</td>
      <td>5</td>
      <td>I</td>
      <td>333</td>
    </tr>
    <tr>
      <th>Response</th>
      <td>383</td>
      <td>4</td>
      <td>Excellent</td>
      <td>208</td>
    </tr>
    <tr>
      <th>Recurred</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>275</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variable levels
# based on the ordered categories
##################################
ordered_cat_cols = thyroid_cancer.select_dtypes(include=["category"]).columns
for col in ordered_cat_cols:
    print(f"Column: {col}")
    print("Absolute Frequencies:")
    print(thyroid_cancer[col].value_counts().reindex(thyroid_cancer[col].cat.categories))
    print("\nNormalized Frequencies:")
    print(thyroid_cancer[col].value_counts(normalize=True).reindex(thyroid_cancer[col].cat.categories))
    print("-" * 50)
   
```

    Column: Gender
    Absolute Frequencies:
    M     71
    F    312
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M    0.185379
    F    0.814621
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Smoking
    Absolute Frequencies:
    No     334
    Yes     49
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.872063
    Yes    0.127937
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Smoking
    Absolute Frequencies:
    No     355
    Yes     28
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.926893
    Yes    0.073107
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Radiotherapy
    Absolute Frequencies:
    No     376
    Yes      7
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.981723
    Yes    0.018277
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Thyroid_Function
    Absolute Frequencies:
    Euthyroid                      332
    Subclinical Hypothyroidism      14
    Subclinical Hyperthyroidism      5
    Clinical Hypothyroidism         12
    Clinical Hyperthyroidism        20
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Euthyroid                      0.866841
    Subclinical Hypothyroidism     0.036554
    Subclinical Hyperthyroidism    0.013055
    Clinical Hypothyroidism        0.031332
    Clinical Hyperthyroidism       0.052219
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Physical_Examination
    Absolute Frequencies:
    Normal                           7
    Single nodular goiter-left      89
    Single nodular goiter-right    140
    Multinodular goiter            140
    Diffuse goiter                   7
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Normal                         0.018277
    Single nodular goiter-left     0.232376
    Single nodular goiter-right    0.365535
    Multinodular goiter            0.365535
    Diffuse goiter                 0.018277
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Adenopathy
    Absolute Frequencies:
    No           277
    Left          17
    Right         48
    Bilateral     32
    Posterior      2
    Extensive      7
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No           0.723238
    Left         0.044386
    Right        0.125326
    Bilateral    0.083551
    Posterior    0.005222
    Extensive    0.018277
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Pathology
    Absolute Frequencies:
    Hurthle Cell       20
    Follicular         28
    Micropapillary     48
    Papillary         287
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Hurthle Cell      0.052219
    Follicular        0.073107
    Micropapillary    0.125326
    Papillary         0.749347
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Focality
    Absolute Frequencies:
    Uni-Focal      247
    Multi-Focal    136
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Uni-Focal      0.644909
    Multi-Focal    0.355091
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Risk
    Absolute Frequencies:
    Low             249
    Intermediate    102
    High             32
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Low             0.650131
    Intermediate    0.266319
    High            0.083551
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: T
    Absolute Frequencies:
    T1a     49
    T1b     43
    T2     151
    T3a     96
    T3b     16
    T4a     20
    T4b      8
    Name: count, dtype: int64
    
    Normalized Frequencies:
    T1a    0.127937
    T1b    0.112272
    T2     0.394256
    T3a    0.250653
    T3b    0.041775
    T4a    0.052219
    T4b    0.020888
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: N
    Absolute Frequencies:
    N0     268
    N1a     22
    N1b     93
    Name: count, dtype: int64
    
    Normalized Frequencies:
    N0     0.699739
    N1a    0.057441
    N1b    0.242820
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: M
    Absolute Frequencies:
    M0    365
    M1     18
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M0    0.953003
    M1    0.046997
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Stage
    Absolute Frequencies:
    I      333
    II      32
    III      4
    IVA      3
    IVB     11
    Name: count, dtype: int64
    
    Normalized Frequencies:
    I      0.869452
    II     0.083551
    III    0.010444
    IVA    0.007833
    IVB    0.028721
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Response
    Absolute Frequencies:
    Excellent                 208
    Structural Incomplete      91
    Biochemical Incomplete     23
    Indeterminate              61
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Excellent                 0.543081
    Structural Incomplete     0.237598
    Biochemical Incomplete    0.060052
    Indeterminate             0.159269
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Recurred
    Absolute Frequencies:
    No     275
    Yes    108
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.718016
    Yes    0.281984
    Name: proportion, dtype: float64
    --------------------------------------------------
    

## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. A total of 19 duplicated rows were identified.
    * In total, 34 observations were affected, consisting of 16 unique occurrences and 19 subsequent duplicates.
    * These 19 duplicates spanned 16 distinct variations, meaning some variations had multiple duplicates.
    * To clean the dataset, all 19 duplicate rows were removed, retaining only the first occurrence of each of the 16 unique variations.
2. No missing data noted for any variable with Null.Count>0 and Fill.Rate<1.0.
3. Low variance observed for 8 variables with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">Hx_Radiotherapy</span>: First.Second.Mode.Ratio = 51.000 (comprised 2 category levels)
    * <span style="color: #FF0000">M</span>: First.Second.Mode.Ratio = 19.222 (comprised 2 category levels)
    * <span style="color: #FF0000">Thyroid_Function</span>: First.Second.Mode.Ratio = 15.650 (comprised 5 category levels)
    * <span style="color: #FF0000">Hx_Smoking</span>: First.Second.Mode.Ratio = 12.000 (comprised 2 category levels)
    * <span style="color: #FF0000">Stage</span>: First.Second.Mode.Ratio = 9.812 (comprised 5 category levels)
    * <span style="color: #FF0000">Smoking</span>: First.Second.Mode.Ratio = 6.428 (comprised 2 category levels)
    * <span style="color: #FF0000">Pathology</span>: First.Second.Mode.Ratio = 6.022 (comprised 4 category levels)
    * <span style="color: #FF0000">Adenopathy</span>: First.Second.Mode.Ratio = 5.375 (comprised 5 category levels)
4. No low variance observed for any variable with Unique.Count.Ratio>10.
5. No high skewness observed for any variable with Skewness>3 or Skewness<(-3).



```python
##################################
# Counting the number of duplicated rows
##################################
thyroid_cancer.duplicated().sum()

```




    np.int64(19)




```python
##################################
# Exploring the duplicated rows
##################################
duplicated_rows = thyroid_cancer[thyroid_cancer.duplicated(keep=False)]
display(duplicated_rows)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>40</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>22</th>
      <td>36</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>32</th>
      <td>36</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>38</th>
      <td>40</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>40</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>61</th>
      <td>35</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>66</th>
      <td>35</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>67</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>69</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>73</th>
      <td>29</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>77</th>
      <td>29</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>106</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>110</th>
      <td>31</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>113</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>115</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>119</th>
      <td>28</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>120</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>121</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>123</th>
      <td>28</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>132</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>136</th>
      <td>21</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>137</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>138</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>142</th>
      <td>42</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>161</th>
      <td>22</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>166</th>
      <td>31</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>168</th>
      <td>21</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>170</th>
      <td>38</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>175</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>178</th>
      <td>38</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>183</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>187</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>189</th>
      <td>42</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>196</th>
      <td>22</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Checking if duplicated rows have identical values across all columns
##################################
num_unique_dup_rows = duplicated_rows.drop_duplicates().shape[0]
num_total_dup_rows = duplicated_rows.shape[0]
if num_unique_dup_rows == 1:
    print("All duplicated rows have the same values across all columns.")
else:
    print(f"There are {num_unique_dup_rows} unique versions among the {num_total_dup_rows} duplicated rows.")
    
```

    There are 16 unique versions among the 35 duplicated rows.
    


```python
##################################
# Counting the unique variations among duplicated rows
##################################
unique_dup_variations = duplicated_rows.drop_duplicates()
variation_counts = duplicated_rows.value_counts().reset_index(name="Count")
print("Unique duplicated row variations and their counts:")
display(variation_counts)
```

    Unique duplicated row variations and their counts:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>29</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>36</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>38</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>40</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>42</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Removing the duplicated rows and
# retaining only the first occurrence
##################################
thyroid_cancer_row_filtered = thyroid_cancer.drop_duplicates(keep="first")
print('Dataset Dimensions: ')
display(thyroid_cancer_row_filtered.shape)

```

    Dataset Dimensions: 
    


    (364, 17)



```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(thyroid_cancer_row_filtered.dtypes)

```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(thyroid_cancer_row_filtered.columns)

```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(thyroid_cancer_row_filtered)] * len(thyroid_cancer_row_filtered.columns))

```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(thyroid_cancer_row_filtered.isna().sum(axis=0))

```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(thyroid_cancer_row_filtered.count())

```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)

```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>int64</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gender</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Smoking</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hx_Smoking</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hx_Radiotherapy</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Thyroid_Function</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Physical_Examination</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adenopathy</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Pathology</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Focality</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Risk</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>T</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>N</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Stage</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Response</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Recurred</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])

```




    0




```python
##################################
# Identifying the rows
# with Fill.Rate < 0.90
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<0.90)]

```


```python
##################################
# Gathering the indices for each observation
##################################
row_index_list = thyroid_cancer_row_filtered.index

```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(thyroid_cancer_row_filtered.columns)] * len(thyroid_cancer_row_filtered))

```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(thyroid_cancer_row_filtered.isna().sum(axis=1))

```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)

```


```python
##################################
# Identifying the rows
# with missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_index_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>359</th>
      <td>378</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>360</th>
      <td>379</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>361</th>
      <td>380</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>362</th>
      <td>381</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>363</th>
      <td>382</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.00
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.00)])

```




    0




```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
thyroid_cancer_numeric = thyroid_cancer_row_filtered.select_dtypes(include='number')

```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = thyroid_cancer_numeric.columns

```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = thyroid_cancer_numeric.min()

```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = thyroid_cancer_numeric.mean()

```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = thyroid_cancer_numeric.median()

```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = thyroid_cancer_numeric.max()

```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[0] for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[1] for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [thyroid_cancer_numeric[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [thyroid_cancer_numeric[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)

```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = thyroid_cancer_numeric.nunique(dropna=True)

```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(thyroid_cancer_numeric)] * len(thyroid_cancer_numeric.columns))

```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)

```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = thyroid_cancer_numeric.skew()

```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = thyroid_cancer_numeric.kurtosis()

```


```python
##################################
# Generating a column quality summary for the numeric column
##################################
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>15</td>
      <td>41.25</td>
      <td>38.0</td>
      <td>82</td>
      <td>31</td>
      <td>27</td>
      <td>21</td>
      <td>13</td>
      <td>1.615385</td>
      <td>65</td>
      <td>364</td>
      <td>0.178571</td>
      <td>0.678269</td>
      <td>-0.359255</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])

```




    0




```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])

```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])

```




    0




```python
##################################
# Formulating the dataset
# with categorical columns only
##################################
thyroid_cancer_categorical = thyroid_cancer_row_filtered.select_dtypes(include='category')

```


```python
##################################
# Gathering the variable names for the categorical column
##################################
categorical_variable_name_list = thyroid_cancer_categorical.columns

```


```python
##################################
# Gathering the first mode values for each categorical column
##################################
categorical_first_mode_list = [thyroid_cancer_row_filtered[x].value_counts().index.tolist()[0] for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the second mode values for each categorical column
##################################
categorical_second_mode_list = [thyroid_cancer_row_filtered[x].value_counts().index.tolist()[1] for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the count of first mode values for each categorical column
##################################
categorical_first_mode_count_list = [thyroid_cancer_categorical[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the count of second mode values for each categorical column
##################################
categorical_second_mode_count_list = [thyroid_cancer_categorical[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the first mode to second mode ratio for each categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)

```


```python
##################################
# Gathering the count of unique values for each categorical column
##################################
categorical_unique_count_list = thyroid_cancer_categorical.nunique(dropna=True)

```


```python
##################################
# Gathering the number of observations for each categorical column
##################################
categorical_row_count_list = list([len(thyroid_cancer_categorical)] * len(thyroid_cancer_categorical.columns))

```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)

```


```python
##################################
# Generating a column quality summary for the categorical columns
##################################
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                    categorical_first_mode_list,
                                                    categorical_second_mode_list,
                                                    categorical_first_mode_count_list,
                                                    categorical_second_mode_count_list,
                                                    categorical_first_second_mode_ratio_list,
                                                    categorical_unique_count_list,
                                                    categorical_row_count_list,
                                                    categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gender</td>
      <td>F</td>
      <td>M</td>
      <td>293</td>
      <td>71</td>
      <td>4.126761</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>315</td>
      <td>49</td>
      <td>6.428571</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hx_Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>336</td>
      <td>28</td>
      <td>12.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hx_Radiotherapy</td>
      <td>No</td>
      <td>Yes</td>
      <td>357</td>
      <td>7</td>
      <td>51.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thyroid_Function</td>
      <td>Euthyroid</td>
      <td>Clinical Hyperthyroidism</td>
      <td>313</td>
      <td>20</td>
      <td>15.650000</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Physical_Examination</td>
      <td>Multinodular goiter</td>
      <td>Single nodular goiter-right</td>
      <td>135</td>
      <td>127</td>
      <td>1.062992</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adenopathy</td>
      <td>No</td>
      <td>Right</td>
      <td>258</td>
      <td>48</td>
      <td>5.375000</td>
      <td>6</td>
      <td>364</td>
      <td>0.016484</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pathology</td>
      <td>Papillary</td>
      <td>Micropapillary</td>
      <td>271</td>
      <td>45</td>
      <td>6.022222</td>
      <td>4</td>
      <td>364</td>
      <td>0.010989</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Focality</td>
      <td>Uni-Focal</td>
      <td>Multi-Focal</td>
      <td>228</td>
      <td>136</td>
      <td>1.676471</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Risk</td>
      <td>Low</td>
      <td>Intermediate</td>
      <td>230</td>
      <td>102</td>
      <td>2.254902</td>
      <td>3</td>
      <td>364</td>
      <td>0.008242</td>
    </tr>
    <tr>
      <th>10</th>
      <td>T</td>
      <td>T2</td>
      <td>T3a</td>
      <td>138</td>
      <td>96</td>
      <td>1.437500</td>
      <td>7</td>
      <td>364</td>
      <td>0.019231</td>
    </tr>
    <tr>
      <th>11</th>
      <td>N</td>
      <td>N0</td>
      <td>N1b</td>
      <td>249</td>
      <td>93</td>
      <td>2.677419</td>
      <td>3</td>
      <td>364</td>
      <td>0.008242</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M</td>
      <td>M0</td>
      <td>M1</td>
      <td>346</td>
      <td>18</td>
      <td>19.222222</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Stage</td>
      <td>I</td>
      <td>II</td>
      <td>314</td>
      <td>32</td>
      <td>9.812500</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Response</td>
      <td>Excellent</td>
      <td>Structural Incomplete</td>
      <td>189</td>
      <td>91</td>
      <td>2.076923</td>
      <td>4</td>
      <td>364</td>
      <td>0.010989</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Recurred</td>
      <td>No</td>
      <td>Yes</td>
      <td>256</td>
      <td>108</td>
      <td>2.370370</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])

```




    8




```python
##################################
# Identifying the categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Hx_Radiotherapy</td>
      <td>No</td>
      <td>Yes</td>
      <td>357</td>
      <td>7</td>
      <td>51.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M</td>
      <td>M0</td>
      <td>M1</td>
      <td>346</td>
      <td>18</td>
      <td>19.222222</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thyroid_Function</td>
      <td>Euthyroid</td>
      <td>Clinical Hyperthyroidism</td>
      <td>313</td>
      <td>20</td>
      <td>15.650000</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hx_Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>336</td>
      <td>28</td>
      <td>12.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Stage</td>
      <td>I</td>
      <td>II</td>
      <td>314</td>
      <td>32</td>
      <td>9.812500</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>315</td>
      <td>49</td>
      <td>6.428571</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pathology</td>
      <td>Papillary</td>
      <td>Micropapillary</td>
      <td>271</td>
      <td>45</td>
      <td>6.022222</td>
      <td>4</td>
      <td>364</td>
      <td>0.010989</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adenopathy</td>
      <td>No</td>
      <td>Right</td>
      <td>258</td>
      <td>48</td>
      <td>5.375000</td>
      <td>6</td>
      <td>364</td>
      <td>0.016484</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])

```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

### 1.4.1 Data Splitting <a class="anchor" id="1.4.1"></a>

1. The baseline dataset (with duplicate rows removed from the original dataset) is comprised of:
    * **364 rows** (observations)
        * **256 Recurred=No**: 70.33%
        * **108 Recurred=Yes**: 29.67%
    * **17 columns** (variables)
        * **1/17 target** (categorical)
             * <span style="color: #FF0000">Recurred</span>
        * **1/17 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
        * **15/17 predictor** (categorical)
             * <span style="color: #FF0000">Gender</span>
             * <span style="color: #FF0000">Smoking</span>
             * <span style="color: #FF0000">Hx_Smoking</span>
             * <span style="color: #FF0000">Hx_Radiotherapy</span>
             * <span style="color: #FF0000">Thyroid_Function</span>
             * <span style="color: #FF0000">Physical_Examination</span>
             * <span style="color: #FF0000">Adenopathy</span>
             * <span style="color: #FF0000">Pathology</span>
             * <span style="color: #FF0000">Focality</span>
             * <span style="color: #FF0000">Risk</span>
             * <span style="color: #FF0000">T</span>
             * <span style="color: #FF0000">N</span>
             * <span style="color: #FF0000">M</span>
             * <span style="color: #FF0000">Stage</span>
             * <span style="color: #FF0000">Response</span>
2. The baseline dataset was divided into three subsets using a fixed random seed:
    * **test data**: 25% of the original data with class stratification applied
    * **train data (initial)**: 75% of the original data with class stratification applied
        * **train data (final)**: 75% of the **train (initial)** data with class stratification applied
        * **validation data**: 25% of the **train (initial)** data with class stratification applied
3. Models were developed from the **train data (final)**. Using the same dataset, a subset of models with optimal hyperparameters were selected, based on cross-validation.
4. Among candidate models with optimal hyperparameters, the final model were selected based on performance on the **validation data**. 
5. Performance of the selected final model (and other candidate models for post-model selection comparison) were evaluated using the **test data**. 
6. The **train data (final)** subset is comprised of:
    * **204 rows** (observations)
        * **143 Recurred=No**: 70.10%
        * **61 Recurred=Yes**: 29.90%
    * **17 columns** (variables)
7. The **validation data** subset is comprised of:
    * **69 rows** (observations)
        * **49 Recurred=No**: 71.01%
        * **20 Recurred=Yes**: 28.98%
    * **17 columns** (variables)
8. The **test data** subset is comprised of:
    * **91 rows** (observations)
        * **64 Recurred=No**: 70.33%
        * **27 Recurred=Yes**: 29.67%
    * **17 columns** (variables)



```python
##################################
# Creating a dataset copy
# of the row filtered data
##################################
thyroid_cancer_baseline = thyroid_cancer_row_filtered.copy()

```


```python
##################################
# Performing a general exploration
# of the baseline dataset
##################################
print('Final Dataset Dimensions: ')
display(thyroid_cancer_baseline.shape)

```

    Final Dataset Dimensions: 
    


    (364, 17)



```python
print('Target Variable Breakdown: ')
thyroid_cancer_breakdown = thyroid_cancer_baseline.groupby('Recurred', observed=True).size().reset_index(name='Count')
thyroid_cancer_breakdown['Percentage'] = (thyroid_cancer_breakdown['Count'] / len(thyroid_cancer_baseline)) * 100
display(thyroid_cancer_breakdown)

```

    Target Variable Breakdown: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recurred</th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>256</td>
      <td>70.32967</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>108</td>
      <td>29.67033</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the train and test data
# from the final dataset
# by applying stratification and
# using a 75-25 ratio
##################################
thyroid_cancer_train_initial, thyroid_cancer_test = train_test_split(thyroid_cancer_baseline, 
                                                               test_size=0.25, 
                                                               stratify=thyroid_cancer_baseline['Recurred'], 
                                                               random_state=88888888)

```


```python
##################################
# Performing a general exploration
# of the initial training dataset
##################################
X_train_initial = thyroid_cancer_train_initial.drop('Recurred', axis = 1)
y_train_initial = thyroid_cancer_train_initial['Recurred']
print('Initial Train Dataset Dimensions: ')
display(X_train_initial.shape)
display(y_train_initial.shape)
print('Initial Train Target Variable Breakdown: ')
display(y_train_initial.value_counts())
print('Initial Train Target Variable Proportion: ')
display(y_train_initial.value_counts(normalize = True))

```

    Initial Train Dataset Dimensions: 
    


    (273, 16)



    (273,)


    Initial Train Target Variable Breakdown: 
    


    Recurred
    No     192
    Yes     81
    Name: count, dtype: int64


    Initial Train Target Variable Proportion: 
    


    Recurred
    No     0.703297
    Yes    0.296703
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the test dataset
##################################
X_test = thyroid_cancer_test.drop('Recurred', axis = 1)
y_test = thyroid_cancer_test['Recurred']
print('Test Dataset Dimensions: ')
display(X_test.shape)
display(y_test.shape)
print('Test Target Variable Breakdown: ')
display(y_test.value_counts())
print('Test Target Variable Proportion: ')
display(y_test.value_counts(normalize = True))

```

    Test Dataset Dimensions: 
    


    (91, 16)



    (91,)


    Test Target Variable Breakdown: 
    


    Recurred
    No     64
    Yes    27
    Name: count, dtype: int64


    Test Target Variable Proportion: 
    


    Recurred
    No     0.703297
    Yes    0.296703
    Name: proportion, dtype: float64



```python
##################################
# Formulating the train and validation data
# from the train dataset
# by applying stratification and
# using a 75-25 ratio
##################################
thyroid_cancer_train, thyroid_cancer_validation = train_test_split(thyroid_cancer_train_initial, 
                                                             test_size=0.25, 
                                                             stratify=thyroid_cancer_train_initial['Recurred'], 
                                                             random_state=88888888)

```


```python
##################################
# Performing a general exploration
# of the final training dataset
##################################
X_train = thyroid_cancer_train.drop('Recurred', axis = 1)
y_train = thyroid_cancer_train['Recurred']
print('Final Train Dataset Dimensions: ')
display(X_train.shape)
display(y_train.shape)
print('Final Train Target Variable Breakdown: ')
display(y_train.value_counts())
print('Final Train Target Variable Proportion: ')
display(y_train.value_counts(normalize = True))

```

    Final Train Dataset Dimensions: 
    


    (204, 16)



    (204,)


    Final Train Target Variable Breakdown: 
    


    Recurred
    No     143
    Yes     61
    Name: count, dtype: int64


    Final Train Target Variable Proportion: 
    


    Recurred
    No     0.70098
    Yes    0.29902
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the validation dataset
##################################
X_validation = thyroid_cancer_validation.drop('Recurred', axis = 1)
y_validation = thyroid_cancer_validation['Recurred']
print('Validation Dataset Dimensions: ')
display(X_validation.shape)
display(y_validation.shape)
print('Validation Target Variable Breakdown: ')
display(y_validation.value_counts())
print('Validation Target Variable Proportion: ')
display(y_validation.value_counts(normalize = True))

```

    Validation Dataset Dimensions: 
    


    (69, 16)



    (69,)


    Validation Target Variable Breakdown: 
    


    Recurred
    No     49
    Yes    20
    Name: count, dtype: int64


    Validation Target Variable Proportion: 
    


    Recurred
    No     0.710145
    Yes    0.289855
    Name: proportion, dtype: float64



```python
##################################
# Saving the training data
# to the DATASETS_FINAL_TRAIN_PATH
# and DATASETS_FINAL_TRAIN_FEATURES_PATH
# and DATASETS_FINAL_TRAIN_TARGET_PATH
##################################
thyroid_cancer_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_PATH, "thyroid_cancer_train.csv"), index=False)
X_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train.csv"), index=False)

```


```python
##################################
# Saving the validation data
# to the DATASETS_FINAL_VALIDATION_PATH
# and DATASETS_FINAL_VALIDATION_FEATURE_PATH
# and DATASETS_FINAL_VALIDATION_TARGET_PATH
##################################
thyroid_cancer_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_PATH, "thyroid_cancer_validation.csv"), index=False)
X_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_FEATURES_PATH, "X_validation.csv"), index=False)
y_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_TARGET_PATH, "y_validation.csv"), index=False)

```


```python
##################################
# Saving the test data
# to the DATASETS_FINAL_TEST_PATH
# and DATASETS_FINAL_TEST_FEATURES_PATH
# and DATASETS_FINAL_TEST_TARGET_PATH
##################################
thyroid_cancer_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_PATH, "thyroid_cancer_test.csv"), index=False)
X_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_FEATURES_PATH, "X_test.csv"), index=False)
y_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_TARGET_PATH, "y_test.csv"), index=False)

```

### 1.4.2 Data Profiling <a class="anchor" id="1.4.2"></a>

1. No distributional anomalies were obseerved for the numeric predictor <span style="color: #FF0000">Age</span>.
2. 9 categorical predictors were observed with categories consisting of too few cases that risk poor generalization and cross-validation issues:
    * <span style="color: #FF0000">Thyroid_Function</span>: 
        * **176** <span style="color: #FF0000">Thyroid_Function=Euthyroid</span>: 86.27%
        * **8** <span style="color: #FF0000">Thyroid_Function=Subclinical Hypothyroidism</span>: 3.92%
        * **3** <span style="color: #FF0000">Thyroid_Function=Subclinical Hyperthyroidism</span>: 1.47%
        * **6** <span style="color: #FF0000">Thyroid_Function=Clinical Hypothyroidism</span>: 2.94%
        * **11** <span style="color: #FF0000">Thyroid_Function=Clinical Hyperthyroidism</span>: 5.39%
    * <span style="color: #FF0000">Physical_Examination</span>:
        * **5** <span style="color: #FF0000">Physical_Examination=Normal</span>: 2.45%
        * **47** <span style="color: #FF0000">Physical_Examination=Single nodular goiter-left</span>: 23.04%
        * **69** <span style="color: #FF0000">Physical_Examination=Single nodular goiter-right</span>: 33.82%
        * **78** <span style="color: #FF0000">Physical_Examination=Multinodular goiter</span>: 38.24%
        * **5** <span style="color: #FF0000">Physical_Examination=Diffuse goiter</span>: 2.45%
    * <span style="color: #FF0000">Adenopathy</span>:
        * **143** <span style="color: #FF0000">Adenopathy=No</span>: 70.09%
        * **8** <span style="color: #FF0000">Adenopathy=Left</span>: 3.92%
        * **26** <span style="color: #FF0000">Adenopathy=Right</span>: 12.75%
        * **1** <span style="color: #FF0000">Adenopathy=Posterior</span>: 0.49%
        * **22** <span style="color: #FF0000">Adenopathy=Bilateral</span>: 10.78%
        * **4** <span style="color: #FF0000">Adenopathy=Extensive</span>: 1.96%
    * <span style="color: #FF0000">Pathology</span>:
        * **11** <span style="color: #FF0000">Pathology=Hurthle Cell</span>: 5.39%
        * **14** <span style="color: #FF0000">Pathology=Follicular</span>: 6.86%
        * **28** <span style="color: #FF0000">Pathology=Micropapillary</span>: 13.73%
        * **151** <span style="color: #FF0000">Pathology=Papillary</span>: 74.02%
    * <span style="color: #FF0000">Risk</span>:
        * **131** <span style="color: #FF0000">Risk=Low</span>: 64.22%
        * **55** <span style="color: #FF0000">Risk=Intermediate</span>: 26.96%
        * **18** <span style="color: #FF0000">Risk=High</span>: 8.82%
    * <span style="color: #FF0000">T</span>:
        * **27** <span style="color: #FF0000">T=T1a</span>: 13.23%
        * **20** <span style="color: #FF0000">T=T1b</span>: 9.80%
        * **79** <span style="color: #FF0000">T=T2</span>: 38.72%
        * **55** <span style="color: #FF0000">T=T3a</span>: 26.96%
        * **6** <span style="color: #FF0000">T=T3b</span>: 2.94%
        * **11** <span style="color: #FF0000">T=T4a</span>: 5.39%
        * **6** <span style="color: #FF0000">T=T4b</span>: 2.94%
    * <span style="color: #FF0000">N</span>:
        * **140** <span style="color: #FF0000">N=N0</span>: 68.63%
        * **13** <span style="color: #FF0000">N=N1a</span>: 6.37%
        * **51** <span style="color: #FF0000">N=N1b</span>: 25.00%
    * <span style="color: #FF0000">Stage</span>:
        * **178** <span style="color: #FF0000">Stage=I</span>: 87.25%
        * **16** <span style="color: #FF0000">Stage=II</span>: 7.84%
        * **1** <span style="color: #FF0000">Stage=III</span>: 0.49%
        * **3** <span style="color: #FF0000">Stage=IVA</span>: 1.47%
        * **6** <span style="color: #FF0000">Stage=IVB</span>: 2.94%
    * <span style="color: #FF0000">Response</span>:
        * **107** <span style="color: #FF0000">Response=Excellent</span>: 52.45%
        * **54** <span style="color: #FF0000">Response=Structural Incomplete</span>: 26.47%
        * **17** <span style="color: #FF0000">Response=Biochemical Incomplete</span>: 8.33%
        * **26** <span style="color: #FF0000">Response=Indeterminate</span>: 12.75%
3. 3 categorical predictors were excluded from the dataset after having been observed with extremely low variance containing categories with very few or almost no variations across observations that may have limited predictive power or drive increased model complexity without performance gains:
    * <span style="color: #FF0000">Hx_Smoking</span>: 
        * **189** <span style="color: #FF0000">Hx_Smoking=No</span>: 92.65%
        * **15** <span style="color: #FF0000">Hx_Smoking=Yes</span>: 7.35%
    * <span style="color: #FF0000">Hx_Radiotherapy</span>: 
        * **199** <span style="color: #FF0000">Hx_Radiotherapy=No</span>: 97.55%
        * **15** <span style="color: #FF0000">Hx_Radiotherapy=Yes</span>: 2.45%
    * <span style="color: #FF0000">M</span>: 
        * **192** <span style="color: #FF0000">M=M0</span>: 94.12%
        * **12** <span style="color: #FF0000">M=M1</span>: 5.88%



```python
##################################
# Segregating the target
# and predictor variables
##################################
thyroid_cancer_train_predictors = thyroid_cancer_train.iloc[:,:-1].columns
thyroid_cancer_train_predictors_numeric = thyroid_cancer_train.iloc[:,:-1].loc[:, thyroid_cancer_train.iloc[:,:-1].columns == 'Age'].columns
thyroid_cancer_train_predictors_categorical = thyroid_cancer_train.iloc[:,:-1].loc[:,thyroid_cancer_train.iloc[:,:-1].columns != 'Age'].columns

```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = thyroid_cancer_train_predictors_numeric

```


```python
##################################
# Segregating the target variable
# and numeric predictors
##################################
histogram_grouping_variable = 'Recurred'
histogram_frequency_variable = numeric_variable_name_list.values[0]

```


```python
##################################
# Comparing the numeric predictors
# grouped by the target variable
##################################
colors = plt.get_cmap('tab10').colors
plt.figure(figsize=(7, 5))
group_no = thyroid_cancer_train[thyroid_cancer_train[histogram_grouping_variable] == 'No'][histogram_frequency_variable]
group_yes = thyroid_cancer_train[thyroid_cancer_train[histogram_grouping_variable] == 'Yes'][histogram_frequency_variable]
plt.hist(group_no, bins=20, alpha=0.5, color=colors[0], label='No', edgecolor='black')
plt.hist(group_yes, bins=20, alpha=0.5, color=colors[1], label='Yes', edgecolor='black')
plt.title(f'{histogram_grouping_variable} Versus {histogram_frequency_variable}')
plt.xlabel(histogram_frequency_variable)
plt.ylabel('Frequency')
plt.legend()
plt.show()

```


    
![png](output_92_0.png)
    



```python
##################################
# Performing a general exploration of the categorical variable levels
# based on the ordered categories
##################################
ordered_cat_cols = thyroid_cancer_train.select_dtypes(include=["category"]).columns
for col in ordered_cat_cols:
    print(f"Column: {col}")
    print("Absolute Frequencies:")
    print(thyroid_cancer_train[col].value_counts().reindex(thyroid_cancer_train[col].cat.categories))
    print("\nNormalized Frequencies:")
    print(thyroid_cancer_train[col].value_counts(normalize=True).reindex(thyroid_cancer_train[col].cat.categories))
    print("-" * 50)
    
```

    Column: Gender
    Absolute Frequencies:
    M     46
    F    158
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M    0.22549
    F    0.77451
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Smoking
    Absolute Frequencies:
    No     172
    Yes     32
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.843137
    Yes    0.156863
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Smoking
    Absolute Frequencies:
    No     189
    Yes     15
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.926471
    Yes    0.073529
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Radiotherapy
    Absolute Frequencies:
    No     199
    Yes      5
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.97549
    Yes    0.02451
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Thyroid_Function
    Absolute Frequencies:
    Euthyroid                      176
    Subclinical Hypothyroidism       8
    Subclinical Hyperthyroidism      3
    Clinical Hypothyroidism          6
    Clinical Hyperthyroidism        11
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Euthyroid                      0.862745
    Subclinical Hypothyroidism     0.039216
    Subclinical Hyperthyroidism    0.014706
    Clinical Hypothyroidism        0.029412
    Clinical Hyperthyroidism       0.053922
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Physical_Examination
    Absolute Frequencies:
    Normal                          5
    Single nodular goiter-left     47
    Single nodular goiter-right    69
    Multinodular goiter            78
    Diffuse goiter                  5
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Normal                         0.024510
    Single nodular goiter-left     0.230392
    Single nodular goiter-right    0.338235
    Multinodular goiter            0.382353
    Diffuse goiter                 0.024510
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Adenopathy
    Absolute Frequencies:
    No           143
    Left           8
    Right         26
    Bilateral     22
    Posterior      1
    Extensive      4
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No           0.700980
    Left         0.039216
    Right        0.127451
    Bilateral    0.107843
    Posterior    0.004902
    Extensive    0.019608
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Pathology
    Absolute Frequencies:
    Hurthle Cell       11
    Follicular         14
    Micropapillary     28
    Papillary         151
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Hurthle Cell      0.053922
    Follicular        0.068627
    Micropapillary    0.137255
    Papillary         0.740196
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Focality
    Absolute Frequencies:
    Uni-Focal      118
    Multi-Focal     86
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Uni-Focal      0.578431
    Multi-Focal    0.421569
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Risk
    Absolute Frequencies:
    Low             131
    Intermediate     55
    High             18
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Low             0.642157
    Intermediate    0.269608
    High            0.088235
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: T
    Absolute Frequencies:
    T1a    27
    T1b    20
    T2     79
    T3a    55
    T3b     6
    T4a    11
    T4b     6
    Name: count, dtype: int64
    
    Normalized Frequencies:
    T1a    0.132353
    T1b    0.098039
    T2     0.387255
    T3a    0.269608
    T3b    0.029412
    T4a    0.053922
    T4b    0.029412
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: N
    Absolute Frequencies:
    N0     140
    N1a     13
    N1b     51
    Name: count, dtype: int64
    
    Normalized Frequencies:
    N0     0.686275
    N1a    0.063725
    N1b    0.250000
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: M
    Absolute Frequencies:
    M0    192
    M1     12
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M0    0.941176
    M1    0.058824
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Stage
    Absolute Frequencies:
    I      178
    II      16
    III      1
    IVA      3
    IVB      6
    Name: count, dtype: int64
    
    Normalized Frequencies:
    I      0.872549
    II     0.078431
    III    0.004902
    IVA    0.014706
    IVB    0.029412
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Response
    Absolute Frequencies:
    Excellent                 107
    Structural Incomplete      54
    Biochemical Incomplete     17
    Indeterminate              26
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Excellent                 0.524510
    Structural Incomplete     0.264706
    Biochemical Incomplete    0.083333
    Indeterminate             0.127451
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Recurred
    Absolute Frequencies:
    No     143
    Yes     61
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.70098
    Yes    0.29902
    Name: proportion, dtype: float64
    --------------------------------------------------
    


```python
##################################
# Segregating the target variable
# and categorical predictors
##################################
proportion_y_variables = thyroid_cancer_train_predictors_categorical
proportion_x_variable = 'Recurred'

```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 5
num_cols = 3

##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 25))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual stacked column plots
# for all categorical columns
##################################
for i, y_variable in enumerate(proportion_y_variables):
    ax = axes[i]
    category_counts = thyroid_cancer_train.groupby([proportion_x_variable, y_variable], observed=True).size().unstack(fill_value=0)
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)
    category_proportions.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{proportion_x_variable} Versus {y_variable}')
    ax.set_xlabel(proportion_x_variable)
    ax.set_ylabel('Proportions')
    ax.legend(loc="lower center")

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()

```


    
![png](output_95_0.png)
    



```python
##################################
# Removing predictors observed with extreme
# near-zero variance and a limited number of levels
##################################
thyroid_cancer_train_column_filtered = thyroid_cancer_train.drop(columns=['Hx_Radiotherapy','M','Hx_Smoking'])
thyroid_cancer_train_column_filtered.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>335</th>
      <td>29</td>
      <td>M</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Extensive</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Intermediate</td>
      <td>T3a</td>
      <td>N1b</td>
      <td>I</td>
      <td>Structural Incomplete</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>201</th>
      <td>25</td>
      <td>F</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>Right</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N1b</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>134</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>Subclinical Hypothyroidism</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>380</th>
      <td>72</td>
      <td>M</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Bilateral</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>IVB</td>
      <td>Structural Incomplete</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



### 1.4.3 Category Aggregration and Encoding <a class="anchor" id="1.4.8"></a>

1. Category aggregation was applied to the previously identified categorical predictors observed with many levels (high-cardinality) containing only a few observations to improve model stability during cross-validation and enhance generalization:
    * <span style="color: #FF0000">Thyroid_Function</span>: 
        * **176** <span style="color: #FF0000">Thyroid_Function=Euthyroid</span>: 86.27%
        * **28** <span style="color: #FF0000">Thyroid_Function=Hypothyroidism or Hyperthyroidism</span>: 13.73%
    * <span style="color: #FF0000">Physical_Examination</span>:
        * **121** <span style="color: #FF0000">Physical_Examination=Normal or Single Nodular Goiter </span>: 59.31%
        * **83** <span style="color: #FF0000">Physical_Examination=Multinodular or Diffuse Goiter</span>: 40.69%
    * <span style="color: #FF0000">Adenopathy</span>:
        * **143** <span style="color: #FF0000">Adenopathy=No</span>: 70.09%
        * **61** <span style="color: #FF0000">Adenopathy=Yes</span>: 29.90%
    * <span style="color: #FF0000">Pathology</span>:
        * **25** <span style="color: #FF0000">Pathology=Non-Papillary </span>: 12.25%
        * **179** <span style="color: #FF0000">Pathology=Papillary</span>: 87.74%
    * <span style="color: #FF0000">Risk</span>:
        * **131** <span style="color: #FF0000">Risk=Low</span>: 64.22%
        * **73** <span style="color: #FF0000">Risk=Intermediate to High</span>: 35.78%
    * <span style="color: #FF0000">T</span>:
        * **126** <span style="color: #FF0000">T=T1 to T2</span>: 61.76%
        * **78** <span style="color: #FF0000">T=T3 to T4b</span>: 38.23%
    * <span style="color: #FF0000">N</span>:
        * **140** <span style="color: #FF0000">N=N0</span>: 68.63%
        * **65** <span style="color: #FF0000">N=N1</span>: 31.37%
    * <span style="color: #FF0000">Stage</span>:
        * **178** <span style="color: #FF0000">Stage=I</span>: 87.25%
        * **26** <span style="color: #FF0000">Stage=II to IVB</span>: 12.75%
    * <span style="color: #FF0000">Response</span>:
        * **107** <span style="color: #FF0000">Response=Excellent</span>: 52.45%
        * **97** <span style="color: #FF0000">Response=Indeterminate or Incomplete</span>: 47.55%



```python
##################################
# Merging small categories into broader groups 
# for certain categorical predictors
# to ensure sufficient representation in statistical models 
# and prevent sparsity issues in cross-validation
##################################
thyroid_cancer_train_column_filtered['Thyroid_Function'] = thyroid_cancer_train_column_filtered['Thyroid_Function'].map(lambda x: 'Euthyroid' if (x in ['Euthyroid'])  else 'Hypothyroidism or Hyperthyroidism').astype('category')
thyroid_cancer_train_column_filtered['Physical_Examination'] = thyroid_cancer_train_column_filtered['Physical_Examination'].map(lambda x: 'Normal or Single Nodular Goiter' if (x in ['Normal', 'Single nodular goiter-left', 'Single nodular goiter-right'])  else 'Multinodular or Diffuse Goiter').astype('category')
thyroid_cancer_train_column_filtered['Adenopathy'] = thyroid_cancer_train_column_filtered['Adenopathy'].map(lambda x: 'No' if x == 'No' else ('Yes' if pd.notna(x) and x != '' else x)).astype('category')
thyroid_cancer_train_column_filtered['Pathology'] = thyroid_cancer_train_column_filtered['Pathology'].map(lambda x: 'Non-Papillary' if (x in ['Hurthle Cell', 'Follicular'])  else 'Papillary').astype('category')
thyroid_cancer_train_column_filtered['Risk'] = thyroid_cancer_train_column_filtered['Risk'].map(lambda x: 'Low' if (x in ['Low'])  else 'Intermediate to High').astype('category')
thyroid_cancer_train_column_filtered['T'] = thyroid_cancer_train_column_filtered['T'].map(lambda x: 'T1 to T2' if (x in ['T1a', 'T1b', 'T2'])  else 'T3 to T4b').astype('category')
thyroid_cancer_train_column_filtered['N'] = thyroid_cancer_train_column_filtered['N'].map(lambda x: 'N0' if (x in ['N0'])  else 'N1').astype('category')
thyroid_cancer_train_column_filtered['Stage'] = thyroid_cancer_train_column_filtered['Stage'].map(lambda x: 'I' if (x in ['I'])  else 'II to IVB').astype('category')
thyroid_cancer_train_column_filtered['Response'] = thyroid_cancer_train_column_filtered['Response'].map(lambda x: 'Indeterminate or Incomplete' if (x in ['Indeterminate', 'Structural Incomplete', 'Biochemical Incomplete'])  else 'Excellent').astype('category')
thyroid_cancer_train_column_filtered.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>335</th>
      <td>29</td>
      <td>M</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>N1</td>
      <td>I</td>
      <td>Indeterminate or Incomplete</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>201</th>
      <td>25</td>
      <td>F</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>Yes</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N1</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>134</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>Hypothyroidism or Hyperthyroidism</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>380</th>
      <td>72</td>
      <td>M</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>N1</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing a general exploration of the categorical variable levels
# based on the ordered categories
##################################
ordered_cat_cols = thyroid_cancer_train_column_filtered.select_dtypes(include=["category"]).columns
for col in ordered_cat_cols:
    print(f"Column: {col}")
    print("Absolute Frequencies:")
    print(thyroid_cancer_train_column_filtered[col].value_counts().reindex(thyroid_cancer_train_column_filtered[col].cat.categories))
    print("\nNormalized Frequencies:")
    print(thyroid_cancer_train_column_filtered[col].value_counts(normalize=True).reindex(thyroid_cancer_train_column_filtered[col].cat.categories))
    print("-" * 50)
    
```

    Column: Gender
    Absolute Frequencies:
    M     46
    F    158
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M    0.22549
    F    0.77451
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Smoking
    Absolute Frequencies:
    No     172
    Yes     32
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.843137
    Yes    0.156863
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Thyroid_Function
    Absolute Frequencies:
    Euthyroid                            176
    Hypothyroidism or Hyperthyroidism     28
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Euthyroid                            0.862745
    Hypothyroidism or Hyperthyroidism    0.137255
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Physical_Examination
    Absolute Frequencies:
    Multinodular or Diffuse Goiter      83
    Normal or Single Nodular Goiter    121
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Multinodular or Diffuse Goiter     0.406863
    Normal or Single Nodular Goiter    0.593137
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Adenopathy
    Absolute Frequencies:
    No     143
    Yes     61
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.70098
    Yes    0.29902
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Pathology
    Absolute Frequencies:
    Non-Papillary     25
    Papillary        179
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Non-Papillary    0.122549
    Papillary        0.877451
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Focality
    Absolute Frequencies:
    Uni-Focal      118
    Multi-Focal     86
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Uni-Focal      0.578431
    Multi-Focal    0.421569
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Risk
    Absolute Frequencies:
    Intermediate to High     73
    Low                     131
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Intermediate to High    0.357843
    Low                     0.642157
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: T
    Absolute Frequencies:
    T1 to T2     126
    T3 to T4b     78
    Name: count, dtype: int64
    
    Normalized Frequencies:
    T1 to T2     0.617647
    T3 to T4b    0.382353
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: N
    Absolute Frequencies:
    N0    140
    N1     64
    Name: count, dtype: int64
    
    Normalized Frequencies:
    N0    0.686275
    N1    0.313725
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Stage
    Absolute Frequencies:
    I            178
    II to IVB     26
    Name: count, dtype: int64
    
    Normalized Frequencies:
    I            0.872549
    II to IVB    0.127451
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Response
    Absolute Frequencies:
    Excellent                      107
    Indeterminate or Incomplete     97
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Excellent                      0.52451
    Indeterminate or Incomplete    0.47549
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Recurred
    Absolute Frequencies:
    No     143
    Yes     61
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.70098
    Yes    0.29902
    Name: proportion, dtype: float64
    --------------------------------------------------
    


```python
##################################
# Segregating the target
# and predictor variables
##################################
thyroid_cancer_train_predictors = thyroid_cancer_train_column_filtered.iloc[:,:-1].columns
thyroid_cancer_train_predictors_numeric = thyroid_cancer_train_column_filtered.iloc[:,:-1].loc[:, thyroid_cancer_train_column_filtered.iloc[:,:-1].columns == 'Age'].columns
thyroid_cancer_train_predictors_categorical = thyroid_cancer_train_column_filtered.iloc[:,:-1].loc[:,thyroid_cancer_train_column_filtered.iloc[:,:-1].columns != 'Age'].columns

```


```python
##################################
# Segregating the target variable
# and categorical predictors
##################################
proportion_y_variables = thyroid_cancer_train_predictors_categorical
proportion_x_variable = 'Recurred'

```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 4
num_cols = 3

##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual stacked column plots
# for all categorical columns
##################################
for i, y_variable in enumerate(proportion_y_variables):
    ax = axes[i]
    category_counts = thyroid_cancer_train_column_filtered.groupby([proportion_x_variable, y_variable], observed=True).size().unstack(fill_value=0)
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)
    category_proportions.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{proportion_x_variable} Versus {y_variable}')
    ax.set_xlabel(proportion_x_variable)
    ax.set_ylabel('Proportions')
    ax.legend(loc="lower center")

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()

```


    
![png](output_102_0.png)
    


### 1.4.4 Outlier and Distributional Shape Analysis <a class="anchor" id="1.4.4"></a>

1. No outliers (Outlier.Count>0, Outlier.Ratio>0.000), high skewness (Skewness>3 or Skewness<(-3)) or abnormal kurtosis (Skewness>2 or Skewness<(-2)) observed for the numeric predictor.
    * <span style="color: #FF0000">Age</span>: Outlier.Count = 0, Outlier.Ratio = 0.000, Skewness = 0.592, Kurtosis = -0.461



```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
thyroid_cancer_train_column_filtered['Age'] = pd.to_numeric(thyroid_cancer_train_column_filtered['Age'])
thyroid_cancer_train_column_filtered_numeric = thyroid_cancer_train_column_filtered.select_dtypes(include='number')
thyroid_cancer_train_column_filtered_numeric = thyroid_cancer_train_column_filtered_numeric.to_frame() if isinstance(thyroid_cancer_train_column_filtered_numeric, pd.Series) else thyroid_cancer_train_column_filtered_numeric

```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = list(thyroid_cancer_train_column_filtered_numeric.columns)

```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = thyroid_cancer_train_column_filtered_numeric.skew()

```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
thyroid_cancer_train_column_filtered_numeric_q1 = thyroid_cancer_train_column_filtered_numeric.quantile(0.25)
thyroid_cancer_train_column_filtered_numeric_q3 = thyroid_cancer_train_column_filtered_numeric.quantile(0.75)
thyroid_cancer_train_column_filtered_numeric_iqr = thyroid_cancer_train_column_filtered_numeric_q3 - thyroid_cancer_train_column_filtered_numeric_q1

```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((thyroid_cancer_train_column_filtered_numeric < (thyroid_cancer_train_column_filtered_numeric_q1 - 1.5 * thyroid_cancer_train_column_filtered_numeric_iqr)) | (thyroid_cancer_train_column_filtered_numeric > (thyroid_cancer_train_column_filtered_numeric_q3 + 1.5 * thyroid_cancer_train_column_filtered_numeric_iqr))).sum() 

```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(thyroid_cancer_train_column_filtered_numeric)] * len(thyroid_cancer_train_column_filtered_numeric.columns))

```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)

```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = thyroid_cancer_train_column_filtered_numeric.skew()

```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = thyroid_cancer_train_column_filtered_numeric.kurtosis()

```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list,
                                                  numeric_skewness_list,
                                                  numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_outlier_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>0.592572</td>
      <td>0</td>
      <td>204</td>
      <td>0.0</td>
      <td>0.592572</td>
      <td>-0.461287</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in thyroid_cancer_train_column_filtered_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=thyroid_cancer_train_column_filtered_numeric, x=column)
    
```


    
![png](output_114_0.png)
    


### 1.4.5 Collinearity <a class="anchor" id="1.4.5"></a>

1. Majority of the predictors reported low (<0.50) to moderate (0.50 to 0.75) correlation.
2. Among pairwise combinations of categorical predictors, high Phi.Coefficient values were noted for:
    * <span style="color: #FF0000">N</span> and <span style="color: #FF0000">Adenopathy</span>: Phi.Coefficient = +0.827
    * <span style="color: #FF0000">N</span> and <span style="color: #FF0000">Risk</span>: Phi.Coefficient = +0.751
    * <span style="color: #FF0000">Adenopathy</span> and <span style="color: #FF0000">Risk</span>: Phi.Coefficient = +0.696
   


```python
##################################
# Creating a dataset copy and
# converting all values to numeric
# for correlation analysis
##################################
pd.set_option('future.no_silent_downcasting', True)
thyroid_cancer_train_correlation = thyroid_cancer_train_column_filtered.copy()
thyroid_cancer_train_correlation_object = thyroid_cancer_train_correlation.iloc[:,1:13].columns
custom_category_orders = {
    'Gender': ['M', 'F'],  
    'Smoking': ['No', 'Yes'],  
    'Thyroid_Function': ['Euthyroid', 'Hypothyroidism or Hyperthyroidism'],  
    'Physical_Examination': ['Normal or Single Nodular Goiter', 'Multinodular or Diffuse Goiter'],  
    'Adenopathy': ['No', 'Yes'],  
    'Pathology': ['Non-Papillary', 'Papillary'],  
    'Focality': ['Uni-Focal', 'Multi-Focal'],  
    'Risk': ['Low', 'Intermediate to High'],  
    'T': ['T1 to T2', 'T3 to T4b'],  
    'N': ['N0', 'N1'],  
    'Stage': ['I', 'II to IVB'],  
    'Response': ['Excellent', 'Indeterminate or Incomplete'] 
}
encoder = OrdinalEncoder(categories=[custom_category_orders[col] for col in thyroid_cancer_train_correlation_object])
thyroid_cancer_train_correlation[thyroid_cancer_train_correlation_object] = encoder.fit_transform(
    thyroid_cancer_train_correlation[thyroid_cancer_train_correlation_object]
)
thyroid_cancer_train_correlation = thyroid_cancer_train_correlation.drop(['Recurred'], axis=1)
display(thyroid_cancer_train_correlation)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>Stage</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>335</th>
      <td>29</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>25</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>134</th>
      <td>51</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>380</th>
      <td>72</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>31</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>231</th>
      <td>21</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>297</th>
      <td>61</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>270</th>
      <td>39</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>67</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>204 rows × 13 columns</p>
</div>



```python
##################################
# Initializing the correlation matrix
##################################
thyroid_cancer_train_correlation_matrix = pd.DataFrame(np.zeros((len(thyroid_cancer_train_correlation.columns), len(thyroid_cancer_train_correlation.columns))),
                                                       columns=thyroid_cancer_train_correlation.columns,
                                                       index=thyroid_cancer_train_correlation.columns)

```


```python
##################################
# Creating an empty correlation matrix
##################################
thyroid_cancer_train_correlation_matrix = pd.DataFrame(
    np.zeros((len(thyroid_cancer_train_correlation.columns), len(thyroid_cancer_train_correlation.columns))),
    index=thyroid_cancer_train_correlation.columns,
    columns=thyroid_cancer_train_correlation.columns
)


##################################
# Calculating different types
# of correlation coefficients
# per variable type
##################################
for i in range(len(thyroid_cancer_train_correlation.columns)):
    for j in range(i, len(thyroid_cancer_train_correlation.columns)):
        if i == j:
            thyroid_cancer_train_correlation_matrix.iloc[i, j] = 1.0  
        else:
            col_i = thyroid_cancer_train_correlation.iloc[:, i]
            col_j = thyroid_cancer_train_correlation.iloc[:, j]

            # Detecting binary variables (assumes binary variables are coded as 0/1)
            is_binary_i = col_i.nunique() == 2
            is_binary_j = col_j.nunique() == 2

            # Computing the Pearson correlation for two continuous variables
            if col_i.dtype in ['int64', 'float64'] and col_j.dtype in ['int64', 'float64']:
                corr = col_i.corr(col_j)

            # Computing the Point-Biserial correlation for continuous and binary variables
            elif (col_i.dtype in ['int64', 'float64'] and is_binary_j) or (col_j.dtype in ['int64', 'float64'] and is_binary_i):
                continuous_var = col_i if col_i.dtype in ['int64', 'float64'] else col_j
                binary_var = col_j if is_binary_j else col_i

                # Convert binary variable to 0/1 (if not already)
                binary_var = binary_var.astype('category').cat.codes
                corr, _ = pointbiserialr(continuous_var, binary_var)

            # Computing the Phi coefficient for two binary variables
            elif is_binary_i and is_binary_j:
                corr = col_i.corr(col_j) 

            # Computing the Cramér's V for two categorical variables (if more than 2 categories)
            else:
                contingency_table = pd.crosstab(col_i, col_j)
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                phi2 = chi2 / n
                r, k = contingency_table.shape
                corr = np.sqrt(phi2 / min(k - 1, r - 1))  # Cramér's V formula

            # Assigning correlation values to the matrix
            thyroid_cancer_train_correlation_matrix.iloc[i, j] = corr
            thyroid_cancer_train_correlation_matrix.iloc[j, i] = corr

# Displaying the correlation matrix
display(thyroid_cancer_train_correlation_matrix)
            
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>Stage</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>-0.194235</td>
      <td>0.384793</td>
      <td>-0.064362</td>
      <td>0.166593</td>
      <td>0.075158</td>
      <td>-0.103521</td>
      <td>0.193520</td>
      <td>0.218433</td>
      <td>0.230384</td>
      <td>0.032385</td>
      <td>0.548657</td>
      <td>0.277894</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>-0.194235</td>
      <td>1.000000</td>
      <td>-0.605869</td>
      <td>-0.023393</td>
      <td>-0.126177</td>
      <td>-0.262486</td>
      <td>0.048746</td>
      <td>-0.204469</td>
      <td>-0.331298</td>
      <td>-0.178901</td>
      <td>-0.292449</td>
      <td>-0.286223</td>
      <td>-0.261361</td>
    </tr>
    <tr>
      <th>Smoking</th>
      <td>0.384793</td>
      <td>-0.605869</td>
      <td>1.000000</td>
      <td>0.023809</td>
      <td>0.136654</td>
      <td>0.307114</td>
      <td>-0.208749</td>
      <td>0.232285</td>
      <td>0.352861</td>
      <td>0.243106</td>
      <td>0.260305</td>
      <td>0.522287</td>
      <td>0.345057</td>
    </tr>
    <tr>
      <th>Thyroid_Function</th>
      <td>-0.064362</td>
      <td>-0.023393</td>
      <td>0.023809</td>
      <td>1.000000</td>
      <td>0.075621</td>
      <td>-0.073821</td>
      <td>-0.068143</td>
      <td>-0.052038</td>
      <td>-0.030299</td>
      <td>-0.079318</td>
      <td>-0.054779</td>
      <td>-0.024290</td>
      <td>-0.151571</td>
    </tr>
    <tr>
      <th>Physical_Examination</th>
      <td>0.166593</td>
      <td>-0.126177</td>
      <td>0.136654</td>
      <td>0.075621</td>
      <td>1.000000</td>
      <td>0.156521</td>
      <td>0.035651</td>
      <td>0.464966</td>
      <td>0.276835</td>
      <td>0.190238</td>
      <td>0.192704</td>
      <td>0.102383</td>
      <td>0.170525</td>
    </tr>
    <tr>
      <th>Adenopathy</th>
      <td>0.075158</td>
      <td>-0.262486</td>
      <td>0.307114</td>
      <td>-0.073821</td>
      <td>0.156521</td>
      <td>1.000000</td>
      <td>0.015525</td>
      <td>0.309718</td>
      <td>0.696240</td>
      <td>0.521653</td>
      <td>0.827536</td>
      <td>0.360418</td>
      <td>0.514449</td>
    </tr>
    <tr>
      <th>Pathology</th>
      <td>-0.103521</td>
      <td>0.048746</td>
      <td>-0.208749</td>
      <td>-0.068143</td>
      <td>0.035651</td>
      <td>0.015525</td>
      <td>1.000000</td>
      <td>-0.104765</td>
      <td>-0.064050</td>
      <td>-0.228898</td>
      <td>0.091596</td>
      <td>-0.081303</td>
      <td>-0.212909</td>
    </tr>
    <tr>
      <th>Focality</th>
      <td>0.193520</td>
      <td>-0.204469</td>
      <td>0.232285</td>
      <td>-0.052038</td>
      <td>0.464966</td>
      <td>0.309718</td>
      <td>-0.104765</td>
      <td>1.000000</td>
      <td>0.439542</td>
      <td>0.451800</td>
      <td>0.364112</td>
      <td>0.269075</td>
      <td>0.379817</td>
    </tr>
    <tr>
      <th>Risk</th>
      <td>0.218433</td>
      <td>-0.331298</td>
      <td>0.352861</td>
      <td>-0.030299</td>
      <td>0.276835</td>
      <td>0.696240</td>
      <td>-0.064050</td>
      <td>0.439542</td>
      <td>1.000000</td>
      <td>0.654179</td>
      <td>0.751465</td>
      <td>0.511978</td>
      <td>0.620217</td>
    </tr>
    <tr>
      <th>T</th>
      <td>0.230384</td>
      <td>-0.178901</td>
      <td>0.243106</td>
      <td>-0.079318</td>
      <td>0.190238</td>
      <td>0.521653</td>
      <td>-0.228898</td>
      <td>0.451800</td>
      <td>0.654179</td>
      <td>1.000000</td>
      <td>0.489771</td>
      <td>0.425255</td>
      <td>0.583975</td>
    </tr>
    <tr>
      <th>N</th>
      <td>0.032385</td>
      <td>-0.292449</td>
      <td>0.260305</td>
      <td>-0.054779</td>
      <td>0.192704</td>
      <td>0.827536</td>
      <td>0.091596</td>
      <td>0.364112</td>
      <td>0.751465</td>
      <td>0.489771</td>
      <td>1.000000</td>
      <td>0.375186</td>
      <td>0.519732</td>
    </tr>
    <tr>
      <th>Stage</th>
      <td>0.548657</td>
      <td>-0.286223</td>
      <td>0.522287</td>
      <td>-0.024290</td>
      <td>0.102383</td>
      <td>0.360418</td>
      <td>-0.081303</td>
      <td>0.269075</td>
      <td>0.511978</td>
      <td>0.425255</td>
      <td>0.375186</td>
      <td>1.000000</td>
      <td>0.371970</td>
    </tr>
    <tr>
      <th>Response</th>
      <td>0.277894</td>
      <td>-0.261361</td>
      <td>0.345057</td>
      <td>-0.151571</td>
      <td>0.170525</td>
      <td>0.514449</td>
      <td>-0.212909</td>
      <td>0.379817</td>
      <td>0.620217</td>
      <td>0.583975</td>
      <td>0.519732</td>
      <td>0.371970</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric and categorical columns
##################################
plt.figure(figsize=(17, 8))
sns.heatmap(thyroid_cancer_train_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

```


    
![png](output_119_0.png)
    


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. Bivariate analysis identified individual predictors with generally positive association to the target variable based on visual inspection.
2. Higher values or higher proportions for the following predictors are associated with the <span style="color: #FF0000">Recurred=Yes</span>category: 
    * <span style="color: #FF0000">Age</span>
    * <span style="color: #FF0000">Gender=M</span>    
    * <span style="color: #FF0000">Smoking=Yes</span>    
    * <span style="color: #FF0000">Physical_Examination=Multinodular or Diffuse Goiter</span>    
    * <span style="color: #FF0000">Adenopathy=Yes</span>
    * <span style="color: #FF0000">Focality=Multi-Focal</span>    
    * <span style="color: #FF0000">Risk=Intermediate to High</span>
    * <span style="color: #FF0000">T=T3 to T4b</span>    
    * <span style="color: #FF0000">N=N1</span>
    * <span style="color: #FF0000">Stage=II to IVB</span>    
    * <span style="color: #FF0000">Response=Indeterminate or Incomplete</span>
3. Proportions for the following predictors are not associated with the <span style="color: #FF0000">Recurred=Yes</span> or <span style="color: #FF0000">Recurred=No</span> categories: 
    * <span style="color: #FF0000">Thyroid_Function</span>
    * <span style="color: #FF0000">Pathology</span>    



```python
##################################
# Segregating the target
# and predictor variables
##################################
thyroid_cancer_train_column_filtered_predictors = thyroid_cancer_train_column_filtered.iloc[:,:-1].columns
thyroid_cancer_train_column_filtered_predictors_numeric = thyroid_cancer_train_column_filtered.iloc[:,:-1].loc[:, thyroid_cancer_train_column_filtered.iloc[:,:-1].columns == 'Age'].columns
thyroid_cancer_train_column_filtered_predictors_categorical = thyroid_cancer_train_column_filtered.iloc[:,:-1].loc[:,thyroid_cancer_train_column_filtered.iloc[:,:-1].columns != 'Age'].columns

```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = thyroid_cancer_train_column_filtered_predictors_numeric

```


```python
##################################
# Segregating the target variable
# and numeric predictors
##################################
boxplot_y_variable = 'Recurred'
boxplot_x_variable = numeric_variable_name_list.values[0]

```


```python
##################################
# Evaluating the numeric predictors
# against the target variable
##################################
plt.figure(figsize=(7, 5))
plt.boxplot([group[boxplot_x_variable] for name, group in thyroid_cancer_train_column_filtered.groupby(boxplot_y_variable, observed=True)])
plt.title(f'{boxplot_y_variable} Versus {boxplot_x_variable}')
plt.xlabel(boxplot_y_variable)
plt.ylabel(boxplot_x_variable)
plt.xticks(range(1, len(thyroid_cancer_train_column_filtered[boxplot_y_variable].unique()) + 1), ['No', 'Yes'])
plt.show()

```


    
![png](output_125_0.png)
    



```python
##################################
# Segregating the target variable
# and categorical predictors
##################################
proportion_y_variables = thyroid_cancer_train_column_filtered_predictors_categorical
proportion_x_variable = 'Recurred'

```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 4
num_cols = 3

##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual stacked column plots
# for all categorical columns
##################################
for i, y_variable in enumerate(proportion_y_variables):
    ax = axes[i]
    category_counts = thyroid_cancer_train_column_filtered.groupby([proportion_x_variable, y_variable], observed=True).size().unstack(fill_value=0)
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)
    category_proportions.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{proportion_x_variable} Versus {y_variable}')
    ax.set_xlabel(proportion_x_variable)
    ax.set_ylabel('Proportions')
    ax.legend(loc="lower center")

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()

```


    
![png](output_127_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictor to the <span style="color: #FF0000">Recurred</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups Yes and No is equal to zero  
    * **Alternative**: Difference in the means between groups Yes and No is not equal to zero   
2. There is sufficient evidence to conclude of a statistically significant difference between the means of the numeric measurements obtained from Yes and No groups of the <span style="color: #FF0000">Recurred</span> target variable in 1 of 1 numeric predictor given its high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Age</span>: T.Test.Statistic=-3.791, T.Test.PValue=0.000
3. The relationship between the categorical predictors to the <span style="color: #FF0000">Recurred</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: The categorical predictor is independent of the categorical target variable 
    * **Alternative**: The categorical predictor is dependent of the categorical target variable    
4. There is sufficient evidence to conclude of a statistically significant relationship between the categories of the categorical predictors and the Yes and No groups of the <span style="color: #FF0000">Recurred</span> target variable in 10 of 12 categorical predictors given their high chisquare statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">Risk</span>: ChiSquare.Test.Statistic=115.387, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">Response</span>: ChiSquare.Test.Statistic=93.015, ChiSquare.Test.PValue=0.000   
    * <span style="color: #FF0000">N</span>: ChiSquare.Test.Statistic=87.380, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">Adenopathy</span>: ChiSquare.Test.Statistic=82.909, ChiSquare.Test.PValue=0.002
    * <span style="color: #FF0000">Stage</span>: ChiSquare.Test.Statistic=58.828, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">T</span>: ChiSquare.Test.Statistic=57.882, ChiSquare.Test.PValue=0.000   
    * <span style="color: #FF0000">Smoking</span>: ChiSquare.Test.Statistic=34.318, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">Gender</span>: ChiSquare.Test.Statistic=29.114, ChiSquare.Test.PValue=0.002
    * <span style="color: #FF0000">Focality</span>: ChiSquare.Test.Statistic=27.017, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">Physical_Examination</span>: ChiSquare.Test.Statistic=5.717, ChiSquare.Test.PValue=0.016



```python
##################################
# Computing the t-test 
# statistic and p-values
# between the target variable
# and numeric predictor columns
##################################
thyroid_cancer_numeric_ttest_target = {}
thyroid_cancer_numeric = thyroid_cancer_train_column_filtered.loc[:,(thyroid_cancer_train_column_filtered.columns == 'Age') | (thyroid_cancer_train_column_filtered.columns == 'Recurred')]
thyroid_cancer_numeric_columns = thyroid_cancer_train_column_filtered_predictors_numeric
for numeric_column in thyroid_cancer_numeric_columns:
    group_0 = thyroid_cancer_numeric[thyroid_cancer_numeric.loc[:,'Recurred']=='No']
    group_1 = thyroid_cancer_numeric[thyroid_cancer_numeric.loc[:,'Recurred']=='Yes']
    thyroid_cancer_numeric_ttest_target['Recurred_' + numeric_column] = stats.ttest_ind(
        group_0[numeric_column], 
        group_1[numeric_column], 
        equal_var=True)

```


```python
##################################
# Formulating the pairwise ttest summary
# between the target variable
# and numeric predictor columns
##################################
thyroid_cancer_numeric_summary = thyroid_cancer_numeric.from_dict(thyroid_cancer_numeric_ttest_target, orient='index')
thyroid_cancer_numeric_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(thyroid_cancer_numeric_summary.sort_values(by=['T.Test.PValue'], ascending=True).head(len(thyroid_cancer_train_column_filtered_predictors_numeric)))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Recurred_Age</th>
      <td>-3.791048</td>
      <td>0.000198</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the chisquare
# statistic and p-values
# between the target variable
# and categorical predictor columns
##################################
thyroid_cancer_categorical_chisquare_target = {}
thyroid_cancer_categorical = thyroid_cancer_train_column_filtered.loc[:,(thyroid_cancer_train_column_filtered.columns != 'Age') | (thyroid_cancer_train_column_filtered.columns == 'Recurred')]
thyroid_cancer_categorical_columns = thyroid_cancer_train_column_filtered_predictors_categorical
for categorical_column in thyroid_cancer_categorical_columns:
    contingency_table = pd.crosstab(thyroid_cancer_categorical[categorical_column], 
                                    thyroid_cancer_categorical['Recurred'])
    thyroid_cancer_categorical_chisquare_target['Recurred_' + categorical_column] = stats.chi2_contingency(
        contingency_table)[0:2]

```


```python
##################################
# Formulating the pairwise chisquare summary
# between the target variable
# and categorical predictor columns
##################################
thyroid_cancer_categorical_summary = thyroid_cancer_categorical.from_dict(thyroid_cancer_categorical_chisquare_target, orient='index')
thyroid_cancer_categorical_summary.columns = ['ChiSquare.Test.Statistic', 'ChiSquare.Test.PValue']
display(thyroid_cancer_categorical_summary.sort_values(by=['ChiSquare.Test.PValue'], ascending=True).head(len(thyroid_cancer_train_column_filtered_predictors_categorical)))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ChiSquare.Test.Statistic</th>
      <th>ChiSquare.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Recurred_Risk</th>
      <td>115.387077</td>
      <td>6.474266e-27</td>
    </tr>
    <tr>
      <th>Recurred_Response</th>
      <td>93.015440</td>
      <td>5.188795e-22</td>
    </tr>
    <tr>
      <th>Recurred_N</th>
      <td>87.380222</td>
      <td>8.954142e-21</td>
    </tr>
    <tr>
      <th>Recurred_Adenopathy</th>
      <td>82.909484</td>
      <td>8.589806e-20</td>
    </tr>
    <tr>
      <th>Recurred_Stage</th>
      <td>58.828665</td>
      <td>1.720169e-14</td>
    </tr>
    <tr>
      <th>Recurred_T</th>
      <td>57.882234</td>
      <td>2.782892e-14</td>
    </tr>
    <tr>
      <th>Recurred_Smoking</th>
      <td>34.318952</td>
      <td>4.678040e-09</td>
    </tr>
    <tr>
      <th>Recurred_Gender</th>
      <td>29.114212</td>
      <td>6.823460e-08</td>
    </tr>
    <tr>
      <th>Recurred_Focality</th>
      <td>27.017885</td>
      <td>2.015816e-07</td>
    </tr>
    <tr>
      <th>Recurred_Physical_Examination</th>
      <td>5.717930</td>
      <td>1.679252e-02</td>
    </tr>
    <tr>
      <th>Recurred_Thyroid_Function</th>
      <td>2.961746</td>
      <td>8.525584e-02</td>
    </tr>
    <tr>
      <th>Recurred_Pathology</th>
      <td>0.891397</td>
      <td>3.450989e-01</td>
    </tr>
  </tbody>
</table>
</div>


## 1.6. Premodelling Data Preparation <a class="anchor" id="1.6"></a>

### 1.6.1 Preprocessed Data Description<a class="anchor" id="1.6.1"></a>

1. A total of 6 of the 16 predictors were excluded from the dataset based on the data preprocessing and exploration findings 
2. There were 3 categorical predictors excluded from the dataset after having been observed with extremely low variance containing categories with very few or almost no variations across observations that may have limited predictive power or drive increased model complexity without performance gains:
    * <span style="color: #FF0000">Hx_Smoking</span>: 
        * **189** <span style="color: #FF0000">Hx_Smoking=No</span>: 92.65%
        * **15** <span style="color: #FF0000">Hx_Smoking=Yes</span>: 7.35%
    * <span style="color: #FF0000">Hx_Radiotherapy</span>: 
        * **199** <span style="color: #FF0000">Hx_Radiotherapy=No</span>: 97.55%
        * **15** <span style="color: #FF0000">Hx_Radiotherapy=Yes</span>: 2.45%
    * <span style="color: #FF0000">M</span>: 
        * **192** <span style="color: #FF0000">M=M0</span>: 94.12%
        * **12** <span style="color: #FF0000">M=M1</span>: 5.88%
2. There was 1 categorical predictor excluded from the dataset after having been observed with high pairwise collinearity (Phi.Coefficient>0.70) with other 2 predictors that might provide redundant information, leading to potential instability in regression models.
    * <span style="color: #FF0000">N</span> was highly associated with <span style="color: #FF0000">Adenopathy</span>: Phi.Coefficient = +0.827
    * <span style="color: #FF0000">N</span> was highly associated with <span style="color: #FF0000">Risk</span>: Phi.Coefficient = +0.751
3. Another 2 categorical predictors were excluded from the dataset for not exhibiting a statistically significant association with the Yes and No groups of the <span style="color: #FF0000">Recurred</span> target variable, indicating weak predictive value.
    * <span style="color: #FF0000">Thyroid_Function</span>: ChiSquare.Test.Statistic=2.962, ChiSquare.Test.PValue=0.085
    * <span style="color: #FF0000">Pathology</span>: ChiSquare.Test.Statistic=0.891, ChiSquare.Test.PValue=0.345   
4. The **preprocessed train data (final)** subset is comprised of:
    * **204 rows** (observations)
        * **143 Recurred=No**: 70.10%
        * **61 Recurred=Yes**: 29.90%
    * **11 columns** (variables)
        * **1/11 target** (categorical)
             * <span style="color: #FF0000">Recurred</span>
        * **1/11 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
        * **9/11 predictor** (categorical)
             * <span style="color: #FF0000">Gender</span>
             * <span style="color: #FF0000">Smoking</span>
             * <span style="color: #FF0000">Physical_Examination</span>
             * <span style="color: #FF0000">Adenopathy</span>
             * <span style="color: #FF0000">Focality</span>
             * <span style="color: #FF0000">Risk</span>
             * <span style="color: #FF0000">T</span>
             * <span style="color: #FF0000">M</span>
             * <span style="color: #FF0000">Stage</span>
             * <span style="color: #FF0000">Response</span>


### 1.6.2 Preprocessing Pipeline Development<a class="anchor" id="1.6.2"></a>

1. A preprocessing pipeline was formulated and applied to the **train data (final)**, **validation data** and **test data** with the following actions:
    * Excluded specified columns noted with low variance, high collinearity and weak predictive power
    * Aggregated categories in multiclass categorical variables into binary levels
    * Converted categorical columns to the appropriate type
    * Set the order of category levels for ordinal encoding during modeling pipeline creation



```python
##################################
# Formulating a preprocessing pipeline
# that removes the specified columns,
# aggregates categories in multiclass categorical variables,
# converts categorical columns to the appropriate type, and
# sets the order of category levels
##################################
def preprocess_dataset(df):
    # Removing the specified columns
    columns_to_remove = ['Hx_Smoking', 'Hx_Radiotherapy', 'M', 'N', 'Thyroid_Function', 'Pathology']
    df = df.drop(columns=columns_to_remove)
    
    # Applying category aggregation
    df['Physical_Examination'] = df['Physical_Examination'].map(
        lambda x: 'Normal or Single Nodular Goiter' if x in ['Normal', 'Single nodular goiter-left', 'Single nodular goiter-right'] 
        else 'Multinodular or Diffuse Goiter').astype('category')
    
    df['Adenopathy'] = df['Adenopathy'].map(
        lambda x: 'No' if x == 'No' else ('Yes' if pd.notna(x) and x != '' else x)).astype('category')
    
    df['Risk'] = df['Risk'].map(
        lambda x: 'Low' if x == 'Low' else 'Intermediate to High').astype('category')
    
    df['T'] = df['T'].map(
        lambda x: 'T1 to T2' if x in ['T1a', 'T1b', 'T2'] else 'T3 to T4b').astype('category')
    
    df['Stage'] = df['Stage'].map(
        lambda x: 'I' if x == 'I' else 'II to IVB').astype('category')
    
    df['Response'] = df['Response'].map(
        lambda x: 'Indeterminate or Incomplete' if x in ['Indeterminate', 'Structural Incomplete', 'Biochemical Incomplete'] 
        else 'Excellent').astype('category')
    
    # Setting category levels
    category_mappings = {
        'Gender': ['M', 'F'],
        'Smoking': ['No', 'Yes'],
        'Physical_Examination': ['Normal or Single Nodular Goiter', 'Multinodular or Diffuse Goiter'],
        'Adenopathy': ['No', 'Yes'],
        'Focality': ['Uni-Focal', 'Multi-Focal'],
        'Risk': ['Low', 'Intermediate to High'],
        'T': ['T1 to T2', 'T3 to T4b'],
        'Stage': ['I', 'II to IVB'],
        'Response': ['Excellent', 'Indeterminate or Incomplete']
    }
    
    for col, categories in category_mappings.items():
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.set_categories(categories, ordered=True)
    
    return df
    
```


```python
##################################
# Applying the preprocessing pipeline
# to the train data
##################################
thyroid_cancer_preprocessed_train = preprocess_dataset(thyroid_cancer_train)
X_preprocessed_train = thyroid_cancer_preprocessed_train.drop('Recurred', axis = 1)
y_preprocessed_train = thyroid_cancer_preprocessed_train['Recurred']
thyroid_cancer_preprocessed_train.to_csv(os.path.join("..", DATASETS_PREPROCESSED_TRAIN_PATH, "thyroid_cancer_preprocessed_train.csv"), index=False)
X_preprocessed_train.to_csv(os.path.join("..", DATASETS_PREPROCESSED_TRAIN_FEATURES_PATH, "X_preprocessed_train.csv"), index=False)
y_preprocessed_train.to_csv(os.path.join("..", DATASETS_PREPROCESSED_TRAIN_TARGET_PATH, "y_preprocessed_train.csv"), index=False)
print('Final Preprocessed Train Dataset Dimensions: ')
display(X_preprocessed_train.shape)
display(y_preprocessed_train.shape)
print('Final Preprocessed Train Target Variable Breakdown: ')
display(y_preprocessed_train.value_counts())
print('Final Preprocessed Train Target Variable Proportion: ')
display(y_preprocessed_train.value_counts(normalize = True))
thyroid_cancer_preprocessed_train.head()

```

    Final Preprocessed Train Dataset Dimensions: 
    


    (204, 10)



    (204,)


    Final Preprocessed Train Target Variable Breakdown: 
    


    Recurred
    No     143
    Yes     61
    Name: count, dtype: int64


    Final Preprocessed Train Target Variable Proportion: 
    


    Recurred
    No     0.70098
    Yes    0.29902
    Name: proportion, dtype: float64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>335</th>
      <td>29</td>
      <td>M</td>
      <td>No</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>I</td>
      <td>Indeterminate or Incomplete</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>201</th>
      <td>25</td>
      <td>F</td>
      <td>No</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>Yes</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>134</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>380</th>
      <td>72</td>
      <td>M</td>
      <td>Yes</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Applying the preprocessing pipeline
# to the validation data
##################################
thyroid_cancer_preprocessed_validation = preprocess_dataset(thyroid_cancer_validation)
X_preprocessed_validation = thyroid_cancer_preprocessed_validation.drop('Recurred', axis = 1)
y_preprocessed_validation = thyroid_cancer_preprocessed_validation['Recurred']
thyroid_cancer_preprocessed_validation.to_csv(os.path.join("..", DATASETS_PREPROCESSED_VALIDATION_PATH, "thyroid_cancer_preprocessed_validation.csv"), index=False)
X_preprocessed_validation.to_csv(os.path.join("..", DATASETS_PREPROCESSED_VALIDATION_FEATURES_PATH, "X_preprocessed_validation.csv"), index=False)
y_preprocessed_validation.to_csv(os.path.join("..", DATASETS_PREPROCESSED_VALIDATION_TARGET_PATH, "y_preprocessed_validation.csv"), index=False)
print('Final Preprocessed Validation Dataset Dimensions: ')
display(X_preprocessed_validation.shape)
display(y_preprocessed_validation.shape)
print('Final Preprocessed Validation Target Variable Breakdown: ')
display(y_preprocessed_validation.value_counts())
print('Final Preprocessed Validation Target Variable Proportion: ')
display(y_preprocessed_validation.value_counts(normalize = True))
thyroid_cancer_preprocessed_validation.head()

```

    Final Preprocessed Validation Dataset Dimensions: 
    


    (69, 10)



    (69,)


    Final Preprocessed Validation Target Variable Breakdown: 
    


    Recurred
    No     49
    Yes    20
    Name: count, dtype: int64


    Final Preprocessed Validation Target Variable Proportion: 
    


    Recurred
    No     0.710145
    Yes    0.289855
    Name: proportion, dtype: float64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49</th>
      <td>29</td>
      <td>F</td>
      <td>No</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>353</th>
      <td>73</td>
      <td>F</td>
      <td>No</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>Yes</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>204</th>
      <td>36</td>
      <td>F</td>
      <td>No</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>Yes</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>283</th>
      <td>30</td>
      <td>F</td>
      <td>No</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>254</th>
      <td>31</td>
      <td>M</td>
      <td>Yes</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T3 to T4b</td>
      <td>I</td>
      <td>Indeterminate or Incomplete</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Applying the preprocessing pipeline
# to the test data
##################################
thyroid_cancer_preprocessed_test = preprocess_dataset(thyroid_cancer_test)
X_preprocessed_test = thyroid_cancer_preprocessed_test.drop('Recurred', axis = 1)
y_preprocessed_test = thyroid_cancer_preprocessed_test['Recurred']
thyroid_cancer_preprocessed_test.to_csv(os.path.join("..", DATASETS_PREPROCESSED_TEST_PATH, "thyroid_cancer_preprocessed_test.csv"), index=False)
X_preprocessed_test.to_csv(os.path.join("..", DATASETS_PREPROCESSED_TEST_FEATURES_PATH, "X_preprocessed_test.csv"), index=False)
y_preprocessed_test.to_csv(os.path.join("..", DATASETS_PREPROCESSED_TEST_TARGET_PATH, "y_preprocessed_test.csv"), index=False)
print('Final Preprocessed Test Dataset Dimensions: ')
display(X_preprocessed_test.shape)
display(y_preprocessed_test.shape)
print('Final Preprocessed Test Target Variable Breakdown: ')
display(y_preprocessed_test.value_counts())
print('Final Preprocessed Test Target Variable Proportion: ')
display(y_preprocessed_test.value_counts(normalize = True))
thyroid_cancer_preprocessed_test.head()

```

    Final Preprocessed Test Dataset Dimensions: 
    


    (91, 10)



    (91,)


    Final Preprocessed Test Target Variable Breakdown: 
    


    Recurred
    No     64
    Yes    27
    Name: count, dtype: int64


    Final Preprocessed Test Target Variable Proportion: 
    


    Recurred
    No     0.703297
    Yes    0.296703
    Name: proportion, dtype: float64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>379</th>
      <td>81</td>
      <td>M</td>
      <td>Yes</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>125</th>
      <td>31</td>
      <td>F</td>
      <td>No</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>Yes</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>286</th>
      <td>58</td>
      <td>F</td>
      <td>No</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>No</td>
    </tr>
    <tr>
      <th>244</th>
      <td>35</td>
      <td>F</td>
      <td>No</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T3 to T4b</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>369</th>
      <td>71</td>
      <td>M</td>
      <td>Yes</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



## 1.7. Bagged Model Development <a class="anchor" id="1.7"></a>

### 1.7.1 Random Forest <a class="anchor" id="1.7.1"></a>


```python
##################################
# Defining the categorical preprocessing parameters
##################################
categorical_features = ['Gender','Smoking','Physical_Examination','Adenopathy','Focality','Risk','T','Stage','Response']
categorical_transformer = OrdinalEncoder()
categorical_preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features)],
                                             remainder='passthrough',
                                             force_int_remainder_cols=False)

```


```python
##################################
# Defining the preprocessing and modeling pipeline parameters
##################################
bagged_rf_pipeline = Pipeline([
    ('categorical_preprocessor', categorical_preprocessor),
    ('bagged_rf_model', RandomForestClassifier(class_weight='balanced', 
                                               random_state=88888888))
])

```


```python
##################################
# Defining hyperparameter grid
##################################
bagged_rf_hyperparameter_grid = {
    'bagged_rf_model__criterion': ['gini', 'entropy'],
    'bagged_rf_model__max_depth': [3, 5],
    'bagged_rf_model__min_samples_leaf': [5, 10],
    'bagged_rf_model__n_estimators': [100, 200]
}

```


```python
##################################
# Defining the cross-validation strategy (5-cycle 5-fold CV)
##################################
cv_strategy = RepeatedStratifiedKFold(n_splits=5, 
                                      n_repeats=5, 
                                      random_state=88888888)

```


```python
##################################
# Performing Grid Search with cross-validation
##################################
bagged_rf_grid_search = GridSearchCV(
    estimator=bagged_rf_pipeline,
    param_grid=bagged_rf_hyperparameter_grid,
    scoring='f1',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)

```


```python
##################################
# Encoding the response variables
# for model evaluation
##################################
y_encoder = OrdinalEncoder()
y_preprocessed_train_encoded = y_encoder.fit_transform(y_preprocessed_train.values.reshape(-1, 1)).ravel()
y_preprocessed_validation_encoded = y_encoder.fit_transform(y_preprocessed_validation.values.reshape(-1, 1)).ravel()

```


```python
##################################
# Fitting GridSearchCV
##################################
bagged_rf_grid_search.fit(X_preprocessed_train, y_preprocessed_train_encoded)

```

    Fitting 25 folds for each of 16 candidates, totalling 400 fits
    




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;categorical_preprocessor&#x27;,
                                        ColumnTransformer(force_int_remainder_cols=False,
                                                          remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;cat&#x27;,
                                                                         OrdinalEncoder(),
                                                                         [&#x27;Gender&#x27;,
                                                                          &#x27;Smoking&#x27;,
                                                                          &#x27;Physical_Examination&#x27;,
                                                                          &#x27;Adenopathy&#x27;,
                                                                          &#x27;Focality&#x27;,
                                                                          &#x27;Risk&#x27;,
                                                                          &#x27;T&#x27;,
                                                                          &#x27;Stage&#x27;,
                                                                          &#x27;Response&#x27;])])),
                                       (&#x27;bagged_rf_model&#x27;,
                                        RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                               random_state=88888888))]),
             n_jobs=-1,
             param_grid={&#x27;bagged_rf_model__criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;bagged_rf_model__max_depth&#x27;: [3, 5],
                         &#x27;bagged_rf_model__min_samples_leaf&#x27;: [5, 10],
                         &#x27;bagged_rf_model__n_estimators&#x27;: [100, 200]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;categorical_preprocessor&#x27;,
                                        ColumnTransformer(force_int_remainder_cols=False,
                                                          remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;cat&#x27;,
                                                                         OrdinalEncoder(),
                                                                         [&#x27;Gender&#x27;,
                                                                          &#x27;Smoking&#x27;,
                                                                          &#x27;Physical_Examination&#x27;,
                                                                          &#x27;Adenopathy&#x27;,
                                                                          &#x27;Focality&#x27;,
                                                                          &#x27;Risk&#x27;,
                                                                          &#x27;T&#x27;,
                                                                          &#x27;Stage&#x27;,
                                                                          &#x27;Response&#x27;])])),
                                       (&#x27;bagged_rf_model&#x27;,
                                        RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                                               random_state=88888888))]),
             n_jobs=-1,
             param_grid={&#x27;bagged_rf_model__criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;bagged_rf_model__max_depth&#x27;: [3, 5],
                         &#x27;bagged_rf_model__min_samples_leaf&#x27;: [5, 10],
                         &#x27;bagged_rf_model__n_estimators&#x27;: [100, 200]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;categorical_preprocessor&#x27;,
                 ColumnTransformer(force_int_remainder_cols=False,
                                   remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;cat&#x27;, OrdinalEncoder(),
                                                  [&#x27;Gender&#x27;, &#x27;Smoking&#x27;,
                                                   &#x27;Physical_Examination&#x27;,
                                                   &#x27;Adenopathy&#x27;, &#x27;Focality&#x27;,
                                                   &#x27;Risk&#x27;, &#x27;T&#x27;, &#x27;Stage&#x27;,
                                                   &#x27;Response&#x27;])])),
                (&#x27;bagged_rf_model&#x27;,
                 RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                        criterion=&#x27;entropy&#x27;, max_depth=5,
                                        min_samples_leaf=5,
                                        random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>categorical_preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for categorical_preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(force_int_remainder_cols=False, remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;cat&#x27;, OrdinalEncoder(),
                                 [&#x27;Gender&#x27;, &#x27;Smoking&#x27;, &#x27;Physical_Examination&#x27;,
                                  &#x27;Adenopathy&#x27;, &#x27;Focality&#x27;, &#x27;Risk&#x27;, &#x27;T&#x27;,
                                  &#x27;Stage&#x27;, &#x27;Response&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Gender&#x27;, &#x27;Smoking&#x27;, &#x27;Physical_Examination&#x27;, &#x27;Adenopathy&#x27;, &#x27;Focality&#x27;, &#x27;Risk&#x27;, &#x27;T&#x27;, &#x27;Stage&#x27;, &#x27;Response&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OrdinalEncoder()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>remainder</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Age&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>passthrough</div></div></label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(class_weight=&#x27;balanced&#x27;, criterion=&#x27;entropy&#x27;,
                       max_depth=5, min_samples_leaf=5, random_state=88888888)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
bagged_rf_optimal = bagged_rf_grid_search.best_estimator_

```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
bagged_rf_optimal_f1_cv = bagged_rf_grid_search.best_score_
bagged_rf_optimal_f1_train = f1_score(y_preprocessed_train_encoded, bagged_rf_optimal.predict(X_preprocessed_train))
bagged_rf_optimal_f1_validation = f1_score(y_preprocessed_validation_encoded, bagged_rf_optimal.predict(X_preprocessed_validation))

```


```python
##################################
# Identifying the optimal model
##################################
print('Best Bagged Model - Random Forest: ')
print(f"Best Random Forest Hyperparameters: {bagged_rf_grid_search.best_params_}")

```

    Best Bagged Model - Random Forest: 
    Best Random Forest Hyperparameters: {'bagged_rf_model__criterion': 'entropy', 'bagged_rf_model__max_depth': 5, 'bagged_rf_model__min_samples_leaf': 5, 'bagged_rf_model__n_estimators': 100}
    


```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {bagged_rf_optimal_f1_cv:.4f}")
print(f"F1 Score on Training Data: {bagged_rf_optimal_f1_train:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_preprocessed_train_encoded, bagged_rf_optimal.predict(X_preprocessed_train)))

```

    F1 Score on Cross-Validated Data: 0.8709
    F1 Score on Training Data: 0.8889
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
             0.0       0.96      0.94      0.95       143
             1.0       0.86      0.92      0.89        61
    
        accuracy                           0.93       204
       macro avg       0.91      0.93      0.92       204
    weighted avg       0.93      0.93      0.93       204
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the train data
##################################
cm_raw = confusion_matrix(y_preprocessed_train_encoded, bagged_rf_optimal.predict(X_preprocessed_train))
cm_normalized = confusion_matrix(y_preprocessed_train_encoded, bagged_rf_optimal.predict(X_preprocessed_train), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Optimal Random Forest Model Performance on Train Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Optimal Random Forest Model Performance on Train Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

```


    
![png](output_153_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
# to assess overfitting optimism
##################################
print(f"F1 Score on Validation Data: {bagged_rf_optimal_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_preprocessed_validation_encoded, bagged_rf_optimal.predict(X_preprocessed_validation)))

```

    F1 Score on Validation Data: 0.8500
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
             0.0       0.94      0.94      0.94        49
             1.0       0.85      0.85      0.85        20
    
        accuracy                           0.91        69
       macro avg       0.89      0.89      0.89        69
    weighted avg       0.91      0.91      0.91        69
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_preprocessed_validation_encoded, bagged_rf_optimal.predict(X_preprocessed_validation))
cm_normalized = confusion_matrix(y_preprocessed_validation_encoded, bagged_rf_optimal.predict(X_preprocessed_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Optimal Random Forest Model Performance on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Optimal Random Forest Model Performance on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

```


    
![png](output_155_0.png)
    



```python
##################################
# Saving the best individual model
# developed from the original training data
################################## 
joblib.dump(bagged_rf_optimal, 
            os.path.join("..", MODELS_PATH, "bagged_model_random_forest_optimal.pkl"))

```




    ['..\\models\\bagged_model_random_forest_optimal.pkl']



### 1.7.2 Extra Trees <a class="anchor" id="1.7.2"></a>


```python
##################################
# Defining the categorical preprocessing parameters
##################################
categorical_features = ['Gender','Smoking','Physical_Examination','Adenopathy','Focality','Risk','T','Stage','Response']
categorical_transformer = OrdinalEncoder()
categorical_preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features)],
                                             remainder='passthrough',
                                             force_int_remainder_cols=False)

```


```python
##################################
# Defining the preprocessing and modeling pipeline parameters
##################################
bagged_et_pipeline = Pipeline([
    ('categorical_preprocessor', categorical_preprocessor),
    ('bagged_et_model', ExtraTreesClassifier(class_weight='balanced', 
                                               random_state=88888888))
])

```


```python
##################################
# Defining hyperparameter grid
##################################
bagged_et_hyperparameter_grid = {
    'bagged_et_model__criterion': ['gini', 'entropy'],
    'bagged_et_model__max_depth': [3, 5],
    'bagged_et_model__min_samples_leaf': [5, 10],
    'bagged_et_model__n_estimators': [100, 200]
}

```


```python
##################################
# Defining the cross-validation strategy (5-cycle 5-fold CV)
##################################
cv_strategy = RepeatedStratifiedKFold(n_splits=5, 
                                      n_repeats=5, 
                                      random_state=88888888)

```


```python
##################################
# Performing Grid Search with cross-validation
##################################
bagged_et_grid_search = GridSearchCV(
    estimator=bagged_et_pipeline,
    param_grid=bagged_et_hyperparameter_grid,
    scoring='f1',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)

```


```python
##################################
# Encoding the response variables
# for model evaluation
##################################
y_encoder = OrdinalEncoder()
y_preprocessed_train_encoded = y_encoder.fit_transform(y_preprocessed_train.values.reshape(-1, 1)).ravel()
y_preprocessed_validation_encoded = y_encoder.fit_transform(y_preprocessed_validation.values.reshape(-1, 1)).ravel()

```


```python
##################################
# Fitting GridSearchCV
##################################
bagged_et_grid_search.fit(X_preprocessed_train, y_preprocessed_train_encoded)

```

    Fitting 25 folds for each of 16 candidates, totalling 400 fits
    




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;categorical_preprocessor&#x27;,
                                        ColumnTransformer(force_int_remainder_cols=False,
                                                          remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;cat&#x27;,
                                                                         OrdinalEncoder(),
                                                                         [&#x27;Gender&#x27;,
                                                                          &#x27;Smoking&#x27;,
                                                                          &#x27;Physical_Examination&#x27;,
                                                                          &#x27;Adenopathy&#x27;,
                                                                          &#x27;Focality&#x27;,
                                                                          &#x27;Risk&#x27;,
                                                                          &#x27;T&#x27;,
                                                                          &#x27;Stage&#x27;,
                                                                          &#x27;Response&#x27;])])),
                                       (&#x27;bagged_et_model&#x27;,
                                        ExtraTreesClassifier(class_weight=&#x27;balanced&#x27;,
                                                             random_state=88888888))]),
             n_jobs=-1,
             param_grid={&#x27;bagged_et_model__criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;bagged_et_model__max_depth&#x27;: [3, 5],
                         &#x27;bagged_et_model__min_samples_leaf&#x27;: [5, 10],
                         &#x27;bagged_et_model__n_estimators&#x27;: [100, 200]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=88888888),
             estimator=Pipeline(steps=[(&#x27;categorical_preprocessor&#x27;,
                                        ColumnTransformer(force_int_remainder_cols=False,
                                                          remainder=&#x27;passthrough&#x27;,
                                                          transformers=[(&#x27;cat&#x27;,
                                                                         OrdinalEncoder(),
                                                                         [&#x27;Gender&#x27;,
                                                                          &#x27;Smoking&#x27;,
                                                                          &#x27;Physical_Examination&#x27;,
                                                                          &#x27;Adenopathy&#x27;,
                                                                          &#x27;Focality&#x27;,
                                                                          &#x27;Risk&#x27;,
                                                                          &#x27;T&#x27;,
                                                                          &#x27;Stage&#x27;,
                                                                          &#x27;Response&#x27;])])),
                                       (&#x27;bagged_et_model&#x27;,
                                        ExtraTreesClassifier(class_weight=&#x27;balanced&#x27;,
                                                             random_state=88888888))]),
             n_jobs=-1,
             param_grid={&#x27;bagged_et_model__criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;bagged_et_model__max_depth&#x27;: [3, 5],
                         &#x27;bagged_et_model__min_samples_leaf&#x27;: [5, 10],
                         &#x27;bagged_et_model__n_estimators&#x27;: [100, 200]},
             scoring=&#x27;f1&#x27;, verbose=1)</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: Pipeline</div></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;categorical_preprocessor&#x27;,
                 ColumnTransformer(force_int_remainder_cols=False,
                                   remainder=&#x27;passthrough&#x27;,
                                   transformers=[(&#x27;cat&#x27;, OrdinalEncoder(),
                                                  [&#x27;Gender&#x27;, &#x27;Smoking&#x27;,
                                                   &#x27;Physical_Examination&#x27;,
                                                   &#x27;Adenopathy&#x27;, &#x27;Focality&#x27;,
                                                   &#x27;Risk&#x27;, &#x27;T&#x27;, &#x27;Stage&#x27;,
                                                   &#x27;Response&#x27;])])),
                (&#x27;bagged_et_model&#x27;,
                 ExtraTreesClassifier(class_weight=&#x27;balanced&#x27;, max_depth=5,
                                      min_samples_leaf=5,
                                      random_state=88888888))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>categorical_preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for categorical_preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(force_int_remainder_cols=False, remainder=&#x27;passthrough&#x27;,
                  transformers=[(&#x27;cat&#x27;, OrdinalEncoder(),
                                 [&#x27;Gender&#x27;, &#x27;Smoking&#x27;, &#x27;Physical_Examination&#x27;,
                                  &#x27;Adenopathy&#x27;, &#x27;Focality&#x27;, &#x27;Risk&#x27;, &#x27;T&#x27;,
                                  &#x27;Stage&#x27;, &#x27;Response&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Gender&#x27;, &#x27;Smoking&#x27;, &#x27;Physical_Examination&#x27;, &#x27;Adenopathy&#x27;, &#x27;Focality&#x27;, &#x27;Risk&#x27;, &#x27;T&#x27;, &#x27;Stage&#x27;, &#x27;Response&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OrdinalEncoder()</pre></div> </div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>remainder</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Age&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>passthrough</div></div></label><div class="sk-toggleable__content fitted"><pre>passthrough</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>ExtraTreesClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html">?<span>Documentation for ExtraTreesClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ExtraTreesClassifier(class_weight=&#x27;balanced&#x27;, max_depth=5, min_samples_leaf=5,
                     random_state=88888888)</pre></div> </div></div></div></div></div></div></div></div></div></div></div>




```python
##################################
# Identifying the best model
##################################
bagged_et_optimal = bagged_et_grid_search.best_estimator_

```


```python
##################################
# Evaluating the F1 scores
# on the training, cross-validation, and validation data
##################################
bagged_et_optimal_f1_cv = bagged_et_grid_search.best_score_
bagged_et_optimal_f1_train = f1_score(y_preprocessed_train_encoded, bagged_et_optimal.predict(X_preprocessed_train))
bagged_et_optimal_f1_validation = f1_score(y_preprocessed_validation_encoded, bagged_et_optimal.predict(X_preprocessed_validation))

```


```python
##################################
# Identifying the optimal model
##################################
print('Best Bagged Model – Extra Trees: ')
print(f"Best Extra Trees Hyperparameters: {bagged_et_grid_search.best_params_}")

```

    Best Bagged Model – Extra Trees: 
    Best Extra Trees Hyperparameters: {'bagged_et_model__criterion': 'gini', 'bagged_et_model__max_depth': 5, 'bagged_et_model__min_samples_leaf': 5, 'bagged_et_model__n_estimators': 100}
    


```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the training and cross-validated data
# to assess overfitting optimism
##################################
print(f"F1 Score on Cross-Validated Data: {bagged_et_optimal_f1_cv:.4f}")
print(f"F1 Score on Training Data: {bagged_et_optimal_f1_train:.4f}")
print("\nClassification Report on Training Data:\n", classification_report(y_preprocessed_train_encoded, bagged_et_optimal.predict(X_preprocessed_train)))

```

    F1 Score on Cross-Validated Data: 0.8774
    F1 Score on Training Data: 0.8819
    
    Classification Report on Training Data:
                   precision    recall  f1-score   support
    
             0.0       0.96      0.93      0.95       143
             1.0       0.85      0.92      0.88        61
    
        accuracy                           0.93       204
       macro avg       0.91      0.92      0.91       204
    weighted avg       0.93      0.93      0.93       204
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the train data
##################################
cm_raw = confusion_matrix(y_preprocessed_train_encoded, bagged_et_optimal.predict(X_preprocessed_train))
cm_normalized = confusion_matrix(y_preprocessed_train_encoded, bagged_et_optimal.predict(X_preprocessed_train), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Optimal Extra Trees Model Performance on Train Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Optimal Extra Trees Model Performance on Train Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

```


    
![png](output_169_0.png)
    



```python
##################################
# Summarizing the F1 score results
# and classification metrics
# on the validation data
# to assess overfitting optimism
##################################
print(f"F1 Score on Validation Data: {bagged_et_optimal_f1_validation:.4f}")
print("\nClassification Report on Validation Data:\n", classification_report(y_preprocessed_validation_encoded, bagged_et_optimal.predict(X_preprocessed_validation)))

```

    F1 Score on Validation Data: 0.8500
    
    Classification Report on Validation Data:
                   precision    recall  f1-score   support
    
             0.0       0.94      0.94      0.94        49
             1.0       0.85      0.85      0.85        20
    
        accuracy                           0.91        69
       macro avg       0.89      0.89      0.89        69
    weighted avg       0.91      0.91      0.91        69
    
    


```python
##################################
# Formulating the raw and normalized
# confusion matrices
# from the validation data
##################################
cm_raw = confusion_matrix(y_preprocessed_validation_encoded, bagged_et_optimal.predict(X_preprocessed_validation))
cm_normalized = confusion_matrix(y_preprocessed_validation_encoded, bagged_et_optimal.predict(X_preprocessed_validation), normalize='true')
fig, ax = plt.subplots(1, 2, figsize=(17, 8))
sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Confusion Matrix (Raw Count): Optimal Extra Trees Model Performance on Validation Data')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix (Normalized): Optimal Extra Trees Model Performance on Validation Data')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

```


    
![png](output_171_0.png)
    



```python
##################################
# Saving the best individual model
# developed from the original training data
################################## 
joblib.dump(bagged_et_optimal, 
            os.path.join("..", MODELS_PATH, "bagged_model_extra_trees_optimal.pkl"))

```




    ['..\\models\\bagged_model_extra_trees_optimal.pkl']



### 1.7.3 Bagged Decision Tree <a class="anchor" id="1.7.3"></a>


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

### 1.7.4 Bagged Logistic Regression <a class="anchor" id="1.7.4"></a>


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

### 1.7.5 Bagged Support Vector Machine <a class="anchor" id="1.7.5"></a>


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

## 1.8. Boosted Model Development <a class="anchor" id="1.8"></a>

### 1.8.1 AdaBoost <a class="anchor" id="1.8.1"></a>

### 1.8.2 Gradient Boosting <a class="anchor" id="1.8.2"></a>

### 1.8.3 XGBoost <a class="anchor" id="1.8.3"></a>

### 1.8.4 Light GBM <a class="anchor" id="1.8.4"></a>

### 1.8.5 CatBoost <a class="anchor" id="1.8.5"></a>

## 1.9. Stacked Model Development <a class="anchor" id="1.9"></a>

### 1.9.1 Base Learner - K-Nearest Neighbors <a class="anchor" id="1.9.1"></a>

### 1.9.2 Base Learner - Support Vector Machine <a class="anchor" id="1.9.2"></a>

### 1.9.3 Base Learner - Ridge Classifier <a class="anchor" id="1.9.3"></a>

### 1.9.4 Base Learner - Neural Network <a class="anchor" id="1.9.4"></a>

### 1.9.5 Base Learner - Decision Tree <a class="anchor" id="1.9.5"></a>

### 1.9.6 Meta Learner - Logistic Regression <a class="anchor" id="1.9.6"></a>

## 1.10. Blended Model Development <a class="anchor" id="1.10"></a>

### 1.10.1 Base Learner - K-Nearest Neighbors <a class="anchor" id="1.10.1"></a>

### 1.10.2 Base Learner - Support Vector Machine <a class="anchor" id="1.10.2"></a>

### 1.10.3 Base Learner - Ridge Classifier <a class="anchor" id="1.10.3"></a>

### 1.10.4 Base Learner - Neural Network <a class="anchor" id="1.10.4"></a>

### 1.10.5 Base Learner - Decision Tree <a class="anchor" id="1.10.5"></a>

### 1.10.6 Meta Learner - Logistic Regression <a class="anchor" id="1.10.6"></a>

## 1.11. Consolidated Summary<a class="anchor" id="1.11"></a>

# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Ensemble Methods for Machine Learning](https://www.manning.com/books/ensemble-methods-for-machine-learning) by Gautam Kunapuli
* **[Book]** [Applied Predictive Modeling](http://appliedpredictivemodeling.com/) by Max Kuhn and Kjell Johnson
* **[Book]** [An Introduction to Statistical Learning](https://www.statlearning.com/) by Gareth James, Daniela Witten, Trevor Hastie and Rob Tibshirani
* **[Book]** [Ensemble Methods: Foundations and Algorithms](https://www.taylorfrancis.com/books/mono/10.1201/b12207/ensemble-methods-zhi-hua-zhou) by Zhi-Hua Zhou
* **[Book]** [Effective XGBoost: Optimizing, Tuning, Understanding, and Deploying Classification Models (Treading on Python)](https://www.taylorfrancis.com/books/mono/10.1201/b12207/ensemble-methods-zhi-hua-zhou) by Matt Harrison, Edward Krueger, Alex Rook, Ronald Legere and Bojan Tunguz
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.image](https://matplotlib.org/stable/api/image_api.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.offsetbox](https://matplotlib.org/stable/api/offsetbox_api.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Python Library API]** [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.svm](https://scikit-learn.org/stable/modules/svm.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model_selection](https://scikit-learn.org/stable/model_selection.html) by Scikit-Learn Team
* **[Python Library API]** [imblearn.over_sampling](https://imbalanced-learn.org/stable/over_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [imblearn.under_sampling](https://imbalanced-learn.org/stable/under_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [StatsModels](https://www.statsmodels.org/stable/index.html) by StatsModels Team
* **[Python Library API]** [SciPy](https://scipy.org/) by SciPy Team
* **[Article]** [Ensemble: Boosting, Bagging, and Stacking Machine Learning](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) by Jason Brownlee (MachineLearningMastery.Com)
* **[Article]** [Stacking Machine Learning: Everything You Need to Know](https://www.machinelearningpro.org/stacking-machine-learning/) by Ada Parker (MachineLearningPro.Org)
* **[Article]** [Ensemble Learning: Bagging, Boosting and Stacking](https://duchesnay.github.io/pystatsml/machine_learning/ensemble_learning.html) by Edouard Duchesnay, Tommy Lofstedt and Feki Younes (Duchesnay.GitHub.IO)
* **[Article]** [Stack Machine Learning Models: Get Better Results](https://developer.ibm.com/articles/stack-machine-learning-models-get-better-results/) by Casper Hansen (Developer.IBM.Com)
* **[Article]** [GradientBoosting vs AdaBoost vs XGBoost vs CatBoost vs LightGBM](https://www.geeksforgeeks.org/gradientboosting-vs-adaboost-vs-xgboost-vs-catboost-vs-lightgbm/) by Geeks for Geeks Team (GeeksForGeeks.Org)
* **[Article]** [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/) by Jason Brownlee (MachineLearningMastery.Com)
* **[Article]** [The Ultimate Guide to AdaBoost Algorithm | What is AdaBoost Algorithm?](https://www.mygreatlearning.com/blog/adaboost-algorithm/) by Ashish Kumar (MyGreatLearning.Com)
* **[Article]** [A Gentle Introduction to Ensemble Learning Algorithms](https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/) by Jason Brownlee (MachineLearningMastery.Com)
* **[Article]** [Ensemble Methods: Elegant Techniques to Produce Improved Machine Learning Results](https://www.toptal.com/machine-learning/ensemble-methods-machine-learning) by Necati Demir (Toptal.Com)
* **[Article]** [The Essential Guide to Ensemble Learning](https://www.v7labs.com/blog/ensemble-learning-guide) by Rohit Kundu (V7Labs.Com)
* **[Article]** [Develop an Intuition for How Ensemble Learning Works](https://machinelearningmastery.com/how-ensemble-learning-works/) by by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Mastering Ensemble Techniques in Machine Learning: Bagging, Boosting, Bayes Optimal Classifier, and Stacking](https://rahuljain788.medium.com/mastering-ensemble-techniques-in-machine-learning-bagging-boosting-bayes-optimal-classifier-and-c1dd8052f53f) by Rahul Jain (Medium)
* **[Article]** [Ensemble Learning: Bagging, Boosting, Stacking](https://ai.plainenglish.io/ml-tutorial-19-ensemble-learning-bagging-boosting-stacking-5a926db20ec5) by Ayşe Kübra Kuyucu (Medium)
* **[Article]** [Ensemble: Boosting, Bagging, and Stacking Machine Learning](https://medium.com/@senozanAleyna/ensemble-boosting-bagging-and-stacking-machine-learning-6a09c31thyroid_cancer778) by Aleyna Şenozan (Medium)
* **[Article]** [Boosting, Stacking, and Bagging for Ensemble Models for Time Series Analysis with Python](https://medium.com/@kylejones_47003/boosting-stacking-and-bagging-for-ensemble-models-for-time-series-analysis-with-python-d74ab9026782) by Kyle Jones (Medium)
* **[Article]** [Different types of Ensemble Techniques — Bagging, Boosting, Stacking, Voting, Blending](https://medium.com/@abhishekjainindore24/different-types-of-ensemble-techniques-bagging-boosting-stacking-voting-blending-b04355a03c93) by Abhishek Jain (Medium)
* **[Article]** [Mastering Ensemble Techniques in Machine Learning: Bagging, Boosting, Bayes Optimal Classifier, and Stacking](https://rahuljain788.medium.com/mastering-ensemble-techniques-in-machine-learning-bagging-boosting-bayes-optimal-classifier-and-c1dd8052f53f) by Rahul Jain (Medium)
* **[Article]** [Understanding Ensemble Methods: Bagging, Boosting, and Stacking](https://divyabhagat.medium.com/understanding-ensemble-methods-bagging-boosting-and-stacking-7683c493ac19) by Divya bhagat (Medium)
* **[Video Tutorial]** [BAGGING vs. BOOSTING vs STACKING in Ensemble Learning | Machine Learning](https://www.youtube.com/watch?v=j9jGLwPa6_E) by Gate Smashers (YouTube)
* **[Video Tutorial]** [What is Ensemble Method in Machine Learning | Bagging | Boosting | Stacking | Voting](https://www.youtube.com/watch?v=obXqwJofQeo) by Data_SPILL (YouTube)
* **[Video Tutorial]** [Ensemble Methods | Bagging | Boosting | Stacking](https://www.youtube.com/watch?v=d7Y8snuu7Rs) by World of Signet (YouTube)
* **[Video Tutorial]** [Ensemble (Boosting, Bagging, and Stacking) in Machine Learning: Easy Explanation for Data Scientists](https://www.youtube.com/watch?v=sN5ZcJLDMaE) by Emma Ding (YouTube)
* **[Video Tutorial]** [Ensemble Learning - Bagging, Boosting, and Stacking explained in 4 minutes!](https://www.youtube.com/watch?v=eLt4a8-316E) by Melissa Van Bussel (YouTube)
* **[Video Tutorial]** [Introduction to Ensemble Learning | Bagging , Boosting & Stacking Techniques](https://www.youtube.com/watch?v=hhRYsyHwn3E) by UncomplicatingTech (YouTube)
* **[Video Tutorial]** [Machine Learning Basics: Ensemble Learning: Bagging, Boosting, Stacking](https://www.youtube.com/watch?v=EbYOnORvrio) by ISSAI_NU (YouTube)
* **[Course]** [DataCamp Python Data Analyst Certificate](https://app.datacamp.com/learn/career-tracks/data-analyst-with-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Python Associate Data Scientist Certificate](https://app.datacamp.com/learn/career-tracks/associate-data-scientist-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Python Data Scientist Certificate](https://app.datacamp.com/learn/career-tracks/data-scientist-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Machine Learning Engineer Certificate](https://app.datacamp.com/learn/career-tracks/machine-learning-engineer) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Machine Learning Scientist Certificate](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) by DataCamp Team (DataCamp)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)



```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

