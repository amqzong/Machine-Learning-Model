# Melanoma Diagnosis Classificaton Model

# Description

Takes in a matrix of features computed for a dataset of skin lesion images, trains and tests an ensemble of machine learning algorithms on the dataset, and outputs the accuracy of the machine learning model by plotting an ROC curve and computing accuracy statistics.

This code was written for a project to create a diagnostic app for melanoma. The code for the app has also been posted in an repository on Github entitled "melanoma-diagnostic-app." The project was conducted from Summer 2016 to Summer 2017 at Rockefeller University in the Laboratory for Investigative Dermatology under the mentorship of Dr. Daniel Gareau. Many of the algorithms used to extract the features for each skin lesion image were developed by Dr. Gareau and are not included here. 

For more information on the project and results, see the paper attached entitled "ProjectSummary_DiagnosticMelanomaTest.pdf."

# Main Files

**Name: main.m**

Description: Takes in a matrix of biomarker features named "Matrix_Out" and runs a bagging classification model on it, by calling an ensemble of machine learning algorithms on the matrix in the helper method "MasterCost" in multiple iterations.  For each image, the probability scores from all the iterations were averaged for the final probability of melanoma for each image. The final probabilities were then compared with the standard histopathologic diagnoses for the 112 images. This comparison was used to plot a receiver operating characteristic (ROC) curve. The optimal threshold for classifying melanoma was determined by choosing the highest specificity at 98% sensitivity.

**Name: MasterCost.m**

Description: Computes the machine learning parameters for 5 algorithms (logistic regression, neural networks, 
support-vector machines, decision trees, and random forest) from a training set of images and tests the 
parameters on a test set in order to evaluate sensitivity, specificity, and overall accuray. 

# Credits

Daniel Gareau (Mentor)

James Krueger (PI)
