#proj_script-DBSCAN-HDBSCAN-TSNE.py 

- Dataset of the code : https://ocslab.hksecurity.net/Datasets/survival-ids

- In this code, it is finding anomaly detection on CAN datasets. 

- This code composed in 5 parts.
    1. function 'get_score' using confusion matrix to define the performance of the model.
    2. function 'dbscan_outlier' using DBSCAN algorithm to detect the anomalies. 
    3. function 'hbscan_outlier' using HDBSCAN algorithm to detect the anomalies.
    4. function 'dbscan_opt_param' using to find optimal parameters for the DBSCAN. 
    5. function 'hdbscan_opt_param' using to find optimal parameters for the HDBSCAN. 
    6. main code for data processing and running functions above.
 
- For the confusion matrix approach, below is each conditions for the matrix.  
    - Calculating the Performance scores. (Malicious: 1, Benign: 0)
      - Benign = 0 & DBSCAN outlier == -1 || HDBSCAN outlier(>threshold) --> False Positive
      - Benign = 0 & DBSCAN outlier != -1 || HDBSCAN inlier(<=threshold) --> True Negative
      - Malicious = 1 & DBSCAN outlier == -1 || HDBSCAN outlier(>threshold) --> True Positive
      - Malicious = 1 & DBSCAN outlier != -1 || HDBSCAN inlier(<=threshold) --> False Negative
    
   
