
# Breast Cancer Classification

This project builds and evaluates machine learning classification models on the Breast Cancer Wisconsin dataset using Python and scikit-learn. The goal is to classify tumors as either malignant or benign based on numerical features computed from digitized images of breast mass samples.

In this project, malignant tumors are treated as the positive class because detecting malignant cases is usually the more important objective in a medical classification task.

## Dataset

This project uses the Breast Cancer Wisconsin dataset from scikit-learn.

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

The dataset contains:

- 569 total samples
- 30 numerical features
- 2 target classes
- 212 malignant samples
- 357 benign samples

## Target Classes 

The original scikit-learn target encoding is:

    0 = malignant 
    1 = benign 

In this project, the labels are converted so that malignant is the positive class: 

    1 = malignant 
    0 = benign 

This makes precision, recall, F1-score, ROC-AUC, and average precision easier to interpret from the perspective of malignant tumor detection.

## Project Structure

    breast-cancer/
    |-- notebooks/
    |-- src/
    |-- results/
    |-- figures/
    |-- .gitignore
    |-- LICENSE
    |-- README.md
    |-- requirements.txt

## Methods

### Data Processing

  - Loading dataset from scikit-learn
  - Train-test split
  - Converting the labels so that malignant becomes the positive class
  - Standardization

### Models

This project compares four classification models:

| Model | Type | Description | 
|---|---|---| 
| SVM Linear | Linear support vector machine | Uses a linear decision boundary to separate malignant and benign tumors.| 
| SVM RBF | Nonlinear support vector machine | Uses a radial basis function kernel to capture nonlinear relationships between features. This model can learn more flexible decision boundaries than Linear SVM. | 
| Logistic Regression | Linear probabilistic classifier | Estimates the probability that a sample belongs to the positive class.| 
| Random Forest | Tree-based ensemble model | Combines many decision trees to improve prediction performance and capture nonlinear relationships.|

### Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` | Proportion of correctly classified samples |
| Precision | `TP / (TP + FP)` | Of the predicted malignant cases, how many were actually malignant |
| Recall | `TP / (TP + FN)` | Of the actual malignant cases, how many were correctly detected |
| ROC-AUC | Area under the ROC curve | Measures ranking quality across thresholds |
| Average Precision | Area under the precision-recall curve | Area under the precision-recall curve |
| F1-score | `2 * (Precision * Recall) / (Precision + Recall)` | Harmonic mean of precision and recall |

Because this is a medical classification task, recall is especially important. A low recall would mean that more malignant tumors are missed.

## Results

| Model | Train Accuracy | Test Accuracy | Train ROC-AUC | Test ROC-AUC | Train AP | Test AP | Test F1 | Test Recall | Test Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM Linear | 0.9846 | 0.9825 | 0.9968 | 0.9937 | 0.9964 | 0.9920 | 0.9762 | 0.9762 | 0.9762 |
| SVM RBF | 0.9868 | 0.9825 | 0.9982 | 0.9977 | 0.9977 | 0.9966 | 0.9762 | 0.9762 | 0.9762 |
| Logistic Regression | 0.9890 | 0.9825 | 0.9975 | 0.9954 | 0.9969 | 0.9940 | 0.9762 | 0.9762 | 0.9762 |
| Random Forest | 1.0000 | 0.9561 | 1.0000 | 0.9939 | 1.0000 | 0.9900 | 0.9398 | 0.9286 | 0.9512 |

## Key Findings

SVM Linear, SVM RBF, and Logistic Regression achieved the same test accuracy, test F1-score, test recall, and test precision. This suggests that these three models produced the same final class predictions on the test set. However, their ROC-AUC and average precision scores are slightly different. This means that although the final predicted class labels are the same, the models rank prediction probabilities differently.

SVM RBF achieved the highest test ROC-AUC and test average precision:

    Test ROC-AUC = 0.9977
    Test AP = 0.9966

This makes SVM RBF the strongest overall model in this project.

Random Forest achieved perfect training performance, but its test accuracy, recall, and F1-score were lower than the other models. This suggests that Random Forest may be overfitting on this dataset.

## ROC Curve

ROC curves show the relationship between the true positive rate and false positive rate across different classification thresholds. The ROC curves of SVM RBF model is shown below.

<p align=center>
<img src="figures/SVM RBF_roc.png" width="500">
</p>


## Precision-Recall Curve

Precision-recall curves are especially useful when the positive class is important. The precision-recall curve of SVM RBF model is shown below.

<p align=center>
<img src="figures/SVM RBF_pr.png" width="500">
</p>

## Coefficient Interpretation

Coefficient plots are generated for linear models such as SVM Linear and Logistic Regression. These plots help interpret which features contribute more strongly to the model's decision. In both SVM Linear and Logistic Regression, the feature with the largest positive coefficient is `worst texture`.

Example figure paths:

    figures/SVM Linear_coef.png
    figures/Logistic Regression_coef.png


## Conclusion

This project compares SVM, Logistic Regression, and Random Forest models for breast cancer classification. The results show that SVM RBF achieved the best overall performance, with the highest test ROC-AUC and average precision.

Because malignant tumors are treated as the positive class, the recall score is especially important. The best-performing models achieved a malignant recall of 0.9762, meaning they correctly detected most malignant cases in the test set.

## How to Run

This project was developed in Google Colab. Run notebooks in order:

    Setup.ipynb
    Breast Cancer Classification.ipynb

## Author

Yi-Heng Tsai
