
# Breast Cancer Classification

This project builds and evaluates machine learning classification models on the Breast Cancer Wisconsin dataset using Python and scikit-learn.

The goal is to classify tumors as either malignant or benign based on numerical features computed from digitized images of breast mass samples.

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
    |
    |-- notebooks/
    |   |-- Setup.ipynb
    |   |-- Breast Cancer Classification.ipynb
    |
    |-- src/
    |   |-- __init__.py
    |   |-- tool.py
    |
    |-- results/
    |   |-- results.csv
    |
    |-- figures/
    |   |-- SVM Linear_cm.png
    |   |-- SVM Linear_roc.png
    |   |-- SVM Linear_pr.png
    |   |-- SVM Linear_coef.png
    |   |-- SVM RBF_cm.png
    |   |-- SVM RBF_roc.png
    |   |-- SVM RBF_pr.png
    |   |-- Logistic Regression_cm.png
    |   |-- Logistic Regression_roc.png
    |   |-- Logistic Regression_pr.png
    |   |-- Logistic Regression_coef.png
    |   |-- Random Forest_cm.png
    |   |-- Random Forest_roc.png
    |   |-- Random Forest_pr.png
    |
    |-- .gitignore
    |-- LICENSE
    |-- README.md
    |-- requirements.txt

## Methods

### 1. Data Loading

The dataset is loaded from scikit-learn.

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

### 2. Train/Test Split

The dataset is split into training and testing sets.

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

The argument `stratify=y` is used to preserve the original class distribution in both the training and testing sets.

### 3. Positive Class Conversion

The original dataset uses:

    0 = malignant
    1 = benign

This project converts the labels so that malignant becomes the positive class:

    y_train = (y_train == 0).astype(int)
    y_test = (y_test == 0).astype(int)

After this conversion:

    1 = malignant
    0 = benign

This is important because metrics such as precision, recall, F1-score, ROC-AUC, and average precision are computed with respect to the positive class.

### 4. Standardization

Standardization is used for models that are sensitive to feature scale, especially SVM and Logistic Regression.

The models are built using scikit-learn pipelines so that preprocessing is applied correctly during training and evaluation.

## Models

This project compares four classification models:

1. SVM Linear
2. SVM RBF
3. Logistic Regression
4. Random Forest

### SVM Linear

SVM Linear uses a linear decision boundary to separate the two classes.

It is useful when the classes are close to linearly separable after feature scaling.

### SVM RBF

SVM RBF uses a radial basis function kernel.

This allows the model to capture nonlinear decision boundaries.

SVM RBF achieved the best overall probability-based performance in this project.

### Logistic Regression

Logistic Regression is a linear classification model that estimates the probability of a sample belonging to the positive class.

It is simple, interpretable, and performs very well on this dataset.

### Random Forest

Random Forest is an ensemble model that combines many decision trees.

It can capture nonlinear relationships, but in this project it shows signs of overfitting because the training performance is perfect while the test performance is lower.

## Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of correctly classified samples |
| ROC-AUC | Measures ranking quality across thresholds |
| Average Precision | Area under the precision-recall curve |
| Precision | Of the predicted malignant cases, how many were actually malignant |
| Recall | Of the actual malignant cases, how many were correctly detected |
| F1-score | Harmonic mean of precision and recall |

Because this is a medical classification task, recall is especially important. A low recall would mean that more malignant tumors are missed.

## Results

The full results are saved in:

    results/results.csv

| Model | Train Accuracy | Test Accuracy | Train ROC-AUC | Test ROC-AUC | Train AP | Test AP | Test F1 | Test Recall | Test Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVM Linear | 0.9846 | 0.9825 | 0.9968 | 0.9937 | 0.9964 | 0.9920 | 0.9762 | 0.9762 | 0.9762 |
| SVM RBF | 0.9868 | 0.9825 | 0.9982 | 0.9977 | 0.9977 | 0.9966 | 0.9762 | 0.9762 | 0.9762 |
| Logistic Regression | 0.9890 | 0.9825 | 0.9975 | 0.9954 | 0.9969 | 0.9940 | 0.9762 | 0.9762 | 0.9762 |
| Random Forest | 1.0000 | 0.9561 | 1.0000 | 0.9939 | 1.0000 | 0.9900 | 0.9398 | 0.9286 | 0.9512 |

## Key Findings

SVM Linear, SVM RBF, and Logistic Regression achieved the same test accuracy, test F1-score, test recall, and test precision.

This suggests that these three models produced the same final class predictions on the test set.

However, their ROC-AUC and average precision scores are slightly different. This means that although the final predicted class labels are the same, the models rank prediction probabilities differently.

SVM RBF achieved the highest test ROC-AUC and test average precision:

    Test ROC-AUC = 0.9977
    Test AP = 0.9966

This makes SVM RBF the strongest overall model in this project.

Random Forest achieved perfect training performance, but its test accuracy, recall, and F1-score were lower than the other models. This suggests that Random Forest may be overfitting on this dataset.

## Best Model

The best overall model is SVM RBF.

It achieved:

- Test Accuracy: 0.9825
- Test ROC-AUC: 0.9977
- Test Average Precision: 0.9966
- Test F1-score: 0.9762
- Test Recall: 0.9762
- Test Precision: 0.9762

SVM RBF performed best because it achieved the strongest probability-based metrics while maintaining the same classification performance as SVM Linear and Logistic Regression.

## Visualizations

This project generates the following visualizations:

- Confusion matrix
- ROC curve
- Precision-recall curve
- Coefficient plots for linear models

The figures are saved in the `figures/` directory.

## Confusion Matrix

Confusion matrices show the number of correct and incorrect predictions for each class.

They help identify whether the model is missing malignant cases or incorrectly labeling benign cases as malignant.

Example figure paths:

    figures/SVM RBF_cm.png
    figures/Logistic Regression_cm.png
    figures/Random Forest_cm.png

## ROC Curve

ROC curves show the relationship between the true positive rate and false positive rate across different classification thresholds.

Example figure paths:

    figures/SVM RBF_roc.png
    figures/Logistic Regression_roc.png
    figures/Random Forest_roc.png

## Precision-Recall Curve

Precision-recall curves are especially useful when the positive class is important.

In this project, the positive class is malignant.

Example figure paths:

    figures/SVM RBF_pr.png
    figures/Logistic Regression_pr.png
    figures/Random Forest_pr.png

## Coefficient Interpretation

Coefficient plots are generated for linear models such as SVM Linear and Logistic Regression.

These plots help interpret which features contribute more strongly to the model's decision.

Example figure paths:

    figures/SVM Linear_coef.png
    figures/Logistic Regression_coef.png


## Conclusion

This project compares SVM, Logistic Regression, and Random Forest models for breast cancer classification.

The results show that SVM RBF achieved the best overall performance, with the highest test ROC-AUC and average precision.

SVM Linear, SVM RBF, and Logistic Regression produced the same final test-set class predictions, resulting in identical test accuracy, recall, precision, and F1-score.

Because malignant tumors are treated as the positive class, the recall score is especially important. The best-performing models achieved a malignant recall of 0.9762, meaning they correctly detected most malignant cases in the test set.


## How to Run

This project was developed in Google Colab. Run notebooks in order:

    Setup.ipynb
    Breast Cancer Classification.ipynb

## Author

Yi-Heng Tsai
