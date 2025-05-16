# 🧠 Brain-Tumor-Classification-Using-MRI-Images

## 📝 Abstract

Brain tumors are a serious medical condition affecting both children and adults, constituting 85 to 90 percent of all primary Central Nervous System (CNS) tumors. Annually, around 11,700 people receive a brain tumor diagnosis, with a 5-year survival rate of approximately 34 percent for men and 36 percent for women. Proper treatment, planning, and accurate diagnostics are crucial to improving patient life expectancy.

This project focuses on automated classification techniques using Deep Learning Algorithms such as Convolutional Neural Network (CNN), Transfer Learning (TL), and Artificial Neural Network (ANN). These techniques offer higher accuracy than manual classification, aiding doctors worldwide in efficient detection and classification of brain tumors.

## 🌐 Context

Brain tumors present complexities in size and location, requiring expertise for accurate analysis. Developing countries often face challenges due to a shortage of skilled doctors and insufficient knowledge about tumors. An automated system on the cloud can address these issues, providing a faster and more accessible solution.

## 🔍 Methodology

1. **Importing Libraries:**  
   - Libraries such as NumPy, Pandas, TensorFlow, and others are imported for data manipulation, visualization, and deep learning model building.

2. **Loading the Dataset:**
   - The dataset containing brain MRI images is loaded into dataframes. File paths and labels are extracted for each image in the dataset.

3. **Data Preprocessing:**
   - Data balance is checked to ensure there is an even distribution of classes.
   - The dataset is split into training, validation, and test sets.
   - ImageDataGenerator is used to preprocess the images and convert dataframes to numpy arrays for model training.

4. **Model Architecture:**
   - Three different convolutional neural network (CNN) models are created using Keras Sequential API:
     - Model 1: Simple CNN model with few layers.
     - Model 2: CNN model with additional layers for increased complexity.
     - Model 3: Transfer learning using a pre-trained VGG16 model with fine-tuning.
     - Model 4: Transfer learning using a pre-trained ResNet50 model with fine-tuning.
     - Model 5: Transfer learning using a pre-trained DenseNet121 model with fine-tuning.

5. **Training the Models:**
   - Each model is compiled using the Adam optimizer and categorical cross-entropy loss.
   - The models are trained on the training dataset for a specified number of epochs, with validation data for evaluation.

6. **Model Performance Analysis:**
   - Training and validation loss and accuracy are plotted over epochs to visualize the models' performance.
   - The best epoch based on validation loss and accuracy is determined for each model.

7. **Model Evaluation:**
   - Each model is evaluated on the training, validation, and test sets to assess its performance.
   - Loss and accuracy metrics are displayed for each set.

8. **Making Predictions:**
   - The trained models are used to make predictions on the test set.
   - Predictions are converted to class labels for evaluation.

9. **Performance Metrics and Visualization:**
   - For each model, performance metrics such as F1-score, precision, recall, and confusion matrix are calculated and displayed.

**Note:** Ensure to update the paths accordingly based on your local machine's directory structure.

**Dataset Download:**
- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**Download Data:** Download the Brain Tumor MRI Dataset from Kaggle mentioned in the dataset section of the project.

**Run the Jupyter Notebook/Google Colab:** Open the provided Jupyter Notebook file and run each cell sequentially. Make sure to update any file paths or configurations as needed for your environment.

**Training and Evaluation:** Train the models using the provided data and evaluate their performance using metrics such as accuracy and loss.

**Interpret Results:** Analyze the model's performance using the visualizations and metrics provided in the notebook.
