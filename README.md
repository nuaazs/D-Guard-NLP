<div align=center>
<h1>D-Guard NLP</h1>
<img src="D-guard.png"  width="380" height="75" />
</div>
<div align=center>
    <img src="https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg" />
<img src="https://img.shields.io/badge/Pytorch-1.10.1-green.svg"  />
<img src="https://img.shields.io/badge/Python-3.9-blue.svg"  />
<img src="https://img.shields.io/badge/Long-Yuan-green.svg"  />
</div>
<div>
<br>
<br>
</div>

## Purpose
D-Guard-NLP is an anti-fraud text classification project that leverages NLP techniques, specifically BERT-based models, for text classification. The main objective of this project is to develop a system capable of identifying fraudulent texts and classifying them into specific categories. The system focuses on the detection and classification of fraud-related phone call texts.

## 🔥 Updates
1. Added CBLoss & Focal Loss: Incorporated CBLoss and Focal Loss functions. These loss functions help in handling class imbalance and focusing on hard examples during the training process.
2. Added Sphere2 & AAM-sofxmax & AM-softmax: In addition to the existing models, we have introduced Sphere2, AAM-softmax, and AM-softmax as alternative architectures for the BERT-based models. These architectures enhance the discriminative power of the models and improve their ability to capture subtle differences in fraudulent text patterns.
3. Added Large Margin Fine-tuning: To enhance the model's ability to separate different classes.

## Project Overview
The D-Guard-NLP project comprises several key components:

1. Data Preprocessing: This step involves cleaning and preprocessing the text data to remove noise and irrelevant information, ensuring high-quality input for the classification models.
2. BERT-based Model Development: The project incorporates the powerful BERT model for training and fine-tuning. BERT-based models are implemented to effectively capture semantic meaning and contextual information in the text data.
3. Text Classification: The trained BERT models are employed to classify text data into either fraud or non-fraud categories. In cases of fraud-related phone call texts, the system further identifies specific fraud categories.
4. Evaluation: The performance of the text classification models is evaluated using appropriate metrics to assess their effectiveness in accurately detecting and classifying fraudulent texts.
5. Deployment: Once the models demonstrate satisfactory performance, the D-Guard-NLP system can be deployed in a production environment, enabling real-time fraud detection and classification of incoming texts.
