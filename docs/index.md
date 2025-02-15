***
# Supervised Learning : Leveraging Ensemble Learning With Bagging, Boosting, Stacking and Blending Approaches

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *February 22, 2025*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Data Preprocessing Pipeline Development](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
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
* <span style="color: #FF0000">Adenopathy</span> - Indication of enlarged lymph nodes in the neck region (Yes | No)
* <span style="color: #FF0000">Pathology</span> - Specific thyroid cancer type as determined by pathology examination of biopsy samples (Follicular | Hurthel Cell | Micropapillary | Papillary)
* <span style="color: #FF0000">Focality</span> - Indication if the cancer is limited to one location or present in multiple locations (Uni-Focal | Multi-Focal)
* <span style="color: #FF0000">Risk</span> - Risk category of the cancer based on various factors, such as tumor size, extent of spread, and histological type (Low | Intermediate | High)
* <span style="color: #FF0000">T</span> - Tumor classification based on its size and extent of invasion into nearby structures (T1a | T1b | T2 | T3a | T3b | T4a | T4b)
* <span style="color: #FF0000">N</span> - Nodal classification indicating the involvement of lymph nodes (N0 | N1a | N1b)
* <span style="color: #FF0000">M</span> - Metastasis classification indicating the presence or absence of distant metastases (M0 | M1)
* <span style="color: #FF0000">Stage</span> - Overall stage of the cancer, typically determined by combining T, N, and M classifications (I | II | III | IVa | IVb)
* <span style="color: #FF0000">Response</span> - Cancer's response to treatment (Biochemical Incomplete | Indetermindate | Excellent | Structural Incomplete)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

### 1.4.1 Data Cleaning <a class="anchor" id="1.4.1"></a>

### 1.4.2 Missing Data Imputation <a class="anchor" id="1.4.2"></a>

### 1.4.3 Outlier Treatment <a class="anchor" id="1.4.3"></a>

### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>

### 1.4.7 Data Encoding <a class="anchor" id="1.4.7"></a>

### 1.4.8 Preprocessed Data Description <a class="anchor" id="1.4.8"></a>

## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

## 1.6. Data Preprocessing Pipeling Development <a class="anchor" id="1.6"></a>

### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>

## 1.7. Bagged Model Development <a class="anchor" id="1.7"></a>

### 1.7.1 Random Forest <a class="anchor" id="1.7.1"></a>

### 1.7.2 Extra Trees <a class="anchor" id="1.7.2"></a>

### 1.7.3 Bagged Decision Tree <a class="anchor" id="1.7.3"></a>

### 1.7.4 Bagged Logistic Regression <a class="anchor" id="1.7.4"></a>

### 1.7.5 Bagged Support Vector Machine <a class="anchor" id="1.7.5"></a>

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
* **[Article]** [Ensemble: Boosting, Bagging, and Stacking Machine Learning](https://medium.com/@senozanAleyna/ensemble-boosting-bagging-and-stacking-machine-learning-6a09c31df778) by Aleyna Şenozan (Medium)
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

