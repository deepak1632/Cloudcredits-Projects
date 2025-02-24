# Spam Email Detection



## Problem Statement

Email communication is an essential part of our daily lives, but spam emails continue to clutter inboxes and pose security risks. This project aims to detect spam emails using machine learning and natural language processing techniques. By preprocessing email text, converting it into numerical features using TF-IDF, and training a Naïve Bayes classifier, the system can accurately classify emails as spam or non-spam (ham). This enables organizations and users to filter out unwanted emails and enhance communication efficiency.

### Steps Followed

1. **Data Collection:**:
- The dataset is provided in a CSV file (Email.csv) that contains email texts and corresponding labels (0 for non-spam, 1 for spam).
  
2. **Data Cleaning and Preprocessing**: 
- Data Cleaning and Preprocessing.
- Convert text to lowercase for uniformity.
- The code checks for a column named either text or message and creates a cleaned version.
            
  
3. **Feature Extraction**: 
- Convert the cleaned text into numerical data using TF-IDF vectorization.
- This step emphasizes important words and minimizes the influence of common but less informative terms.

4. **Data Splitting**: 
- Split the data into training (80%) and testing (20%) sets to evaluate model performance on unseen data.
  
5. **Model Training**: 
- Train a Multinomial Naïve Bayes classifier on the vectorized training data.

6. **Model Evaluation** : 
- Evaluate the classifier using accuracy, precision, recall, and generate a detailed classification report.


### Methodology
#### Data Preprocessing

- **Text Cleaning** : A custom clean_text function uses regular expressions to remove unwanted characters (HTML tags, punctuation, numbers) and normalize the text.

- **Column Verification** : The script ensures that the dataset contains a column for the email content (either text or message) and a column named label for the classification target.

#### Feature Extraction

- **TF-IDF Vectorization** : The cleaned email texts are transformed into numerical feature vectors using the TF-IDF technique, which weighs each word by its importance in the dataset.

#### Model Training and Evaluation
  
- **Profit Margin %** = DIVIDE(SUM(Sales_Data[Profit]), SUM(Sales_Data[Revenue])) * 100
  
- **Revenue Contribution %** = DIVIDE(SUM(Sales_Data[Revenue]), CALCULATE(SUM(Sales_Data[Revenue]), ALL(Sales_Data[Product_Category])))

### How It Detects Spam Emails

1. **Data Cleaning**: The clean_text function normalizes email content by removing HTML tags, punctuation, and extraneous characters, ensuring that the model focuses on the actual text.
  
2. **Feature Extraction with TF-IDF**: The cleaned text is converted into numerical vectors using TF-IDF. This method assigns a weight to each word based on its frequency in the document and its inverse frequency across all documents, highlighting terms that are most indicative of spam.

3. **Naïve Bayes Classification**: The Naïve Bayes classifier learns from the training data by calculating the probability of each word appearing in spam and non-spam emails. When a new email is processed, the classifier uses these probabilities to determine whether the email is likely to be spam.

4. **Model Evaluation**: Evaluation metrics such as accuracy, precision, and recall provide quantitative measures of how well the model distinguishes between spam and non-spam emails.

### Insights

- **Preprocessing Effectiveness**:Proper cleaning of the email text is crucial for effective spam detection.

- **TF-IDF Utility**:TF-IDF transformation highlights important words that differentiate spam from non-spam, improving classifier performance.

- **Naïve Bayes Efficiency**: The Multinomial Naïve Bayes classifier is efficient and performs well in text classification tasks, making it suitable for spam detection.


### Conclusion

This project successfully demonstrates a complete pipeline for spam email detection using machine learning. By integrating data cleaning, TF-IDF vectorization, and a Naïve Bayes classifier, the system can accurately identify spam emails. The approach provides a robust foundation for further improvements and real-time spam filtering applications.

### Future Work

- **Algorithm Exploration :**  Experiment with alternative classifiers (e.g., SVM, Random Forest) or ensemble methods to further improve performance.

- **Advanced NLP Techniques :**  Incorporate deep learning models or word embeddings for richer feature representation.

- **Real-Time Application :**  Develop a user-friendly interface or API for real-time spam detection and email filtering.



