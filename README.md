# The Effect of Conditioning of Trigonometric Transformations of Dates with Meteorological Data in Forest Fires Prediction : An Experimental Study

This repository houses the implementation of USC's EE559 Course Project by Sarthak Kumar Maharana and Shoumik Nandi.

## Abstract 
In this work, we study the effects of conditioning a trigonometric transformation of dates with meteorological data, that would aid in predicting 
the occurrence of forest fires in Algeria. We pose this as a binary classification problem. At first, we experiment by computing statistical 
parameters of 'Temperature', 'Wind Speed (Ws)', 'Humidity (RH)', and 'Rain', by sliding windows of lengths 2, 3, and 7 days. We also condition 
the "encoded" dates, especially the cosine of the days, with the best features obtained. Experimental results reveal that an optimal sliding window 
of size 3 days, is beneficial to obtain good predictions. We train various machine learning models to study the effect of our hypothesis. 
Support Vector Machines (SVM), with a linear kernel and a regularisation parameter of value 5, without any conditioning, gave the best results with 
a test accuracy of 0.916 and F1 score of 0.893, and a relative improvement of 5.84% in accuracy and 7.2% in F1 score, over a nearest-means 
baseline classifier. The corresponding SVM model conditioned with the transformed dates  gave a relative improvement of 3.92% in accuracy, 
over the baseline.

## Installation 
    $ git clone https://github.com/sarthaxxxxx/EE559_project.git
    $ cd EE559_project/
    $ pip install requirements.txt
    
## Running of the scripts
To run and reproduce the results of the project, please run 'run.sh', in the terminal, as:

    $ bash run.sh
    
 However, if you wish to make certain changes, please do so at configs/cfg.py. Everything is self-explanatory. 
 

 Contacts
 --------
 Sarthak Kumar Maharana - email: maharana@usc.edu \
 Shoumik Nandi - email: shoumikn@usc.edu
 
 
