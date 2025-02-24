# Breast Cancer Prediction

## Problem Statement

Breast cancer remains one of the leading causes of mortality among women. Early detection is critical for effective treatment planning. This project develops a machine learning model that :

- Classifies tumors as benign or malignant.
- Calculates estimated operation charges for malignant cases.

### Steps Followed

1. **Data Loading :**:
- Load the dataset from a CSV file using Pandas.
- Print a preview and information about the dataset to validate its structure.
  
2. **Target Mapping**: 
- Check for the diagnosis column.
- Convert the diagnosis values: 'B' → 0 (benign) and 'M' → 1 (malignant), and create a new target column.
            
3. **Feature Selection**: 
- Remove non-feature columns such as id, diagnosis, and Unnamed: 32 to retain only relevant numeric features.

4. **Data Splitting**: 
- Divide the dataset into training (80%) and testing (20%) sets to enable model evaluation on unseen data.
  
5. **Model Training**: 
- Train an SVM classifier with an RBF kernel on the training data.

6. **Model Evaluation** : 
- Make predictions on the test set and evaluate the model using accuracy, precision, recall, and generate a classification report.


### Methodology
#### Support Vector Machine (SVM)

- **Principle** : SVM finds an optimal hyperplane that separates the classes in the feature space using support vectors, thereby maximizing the margin.

- **Kernel Trick** : The RBF (Radial Basis Function) kernel is used to map input features into a higher-dimensional space, allowing the model to handle non-linearly separable data.

#### Feature Extraction

- **TF-IDF Vectorization** : The cleaned email texts are transformed into numerical feature vectors using the TF-IDF technique, which weighs each word by its importance in the dataset.

#### Operation Charge Calculation
  
- **Business Logic** = Each email (or in this case, each tumor diagnosis) predicted as malignant incurs a fixed operation charge of $20,000.
- Eg : Example: If the model predicts 10 malignant cases, the total estimated operation charge is $200,000.
  
###  Evaluation Metrics

1. **Accuracy**: The overall correctness of the model.
  
2. **Precision**: The proportion of predicted malignant cases that are actually malignant.

3. **Recall**: The proportion of actual malignant cases that were correctly identified.

4. **Classification Report**: A detailed breakdown of the performance metrics for both classes.

### Insights

- **Model Performance**: The evaluation metrics (accuracy, precision, recall, and classification report) indicate the effectiveness of the SVM classifier in distinguishing between benign and malignant cases.

- **Operation Charge Estimation** : The system calculates the total and per-case operation charges for malignant predictions, providing valuable financial insights for treatment planning.

### Conclusion

This project demonstrates an end-to-end pipeline for breast cancer prediction and cost estimation using SVM. The combination of data preprocessing, effective feature selection, and a robust classifier results in a system that not only predicts tumor malignancy with high accuracy but also provides an estimation of operation charges. This integrated approach supports both clinical decision-making and budget planning.

### Future Work

- **Algorithm Enhancement :** Experiment with alternative models such as Random Forest or ensemble methods to further improve prediction accuracy.

- **Feature Engineering:**  Explore additional feature extraction and dimensionality reduction techniques.

- **Deployment :** Develop a user-friendly web interface or integrate the model into clinical systems for real-time predictions and cost analysis.

### References

- Breast Cancer Wisconsin (Diagnostic) Data Set.
- Scikit-learn Documentation on SVM.
- Relevant research articles and textbooks on machine learning and medical diagnosis.



