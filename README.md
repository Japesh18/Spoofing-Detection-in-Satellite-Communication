# Spoofing-Detection-in-Satellite-Communication
This is the official repository for the research project successfully completed by me under the guidance of Dr. Brahmjit Singh, National Institute of Technology, Kurukshetra. This project undertakes the detection of spoofed or false signals received in the receiver of a GNSS (Global Navigation Satellite System).

**Problem Statement**
The problem of spoofing (or false) signals in Satellite communication is widespread and it hinders the ability to communicate effectively. Most importantly, necessary information such as Military communication or any information whcih is of utmost importance can be altered through spoofing. So it becomes necessary to filter out the original information from the spoofed information.

**Intial Study**
For solving this problem, we thought to develop a machine learning or deep learning to model that could classify whether a given information is genuine or spoofed. Initial models used XGBoostClassifier to solve this problem. But XGBoostClassifier was found to be not efficient to study the complex data. The data used was TEXBAT dataset. The complete data was stored in 10 binary files, with 2 binary files containing genuine (or cleaned data) and the rest 8 containing spoofed data. The files were of size exceeding 440 GBs. At first, it was found out about the data contained in the files and on what features the samples were based. The features used were Mean, Standard Deviation, Kurtosis and Skewness of Amplitude with Mean and Standard Deviation of the phase of the incoming signals with them being classified as genuine or spoofed.

**Data Collection**
Since the files exceeded the hardware specifications available than needed, the files were read in chunks to prevent RAM overloading.For research purpose, the data was extracted 5% of each spoofed file. To maintain class balance 20% of genuine data from each file was extracted. The code for genuine samples extraction was "20 Percent Of Clean Data" and the code for spoofed samples extraction was "Feature Extraction".

**Data Preprocessing**
The data was preprocessed and split into training, validation and testing datasets. Z-Score normalisation was applied on training dataset and then, the normalisation variables were used to normalise the validation and testing datasets from training dataset.

**Model Building**
Initially, machine learning models were tried. Since, it was a classification problem, starting from Logistic Regression many algorithms till XGBoostClassifier were tried. These models failed to give a sufficient accuracy. Then, deep learning models were thought. While trying deep learning models, we built a Feed Forward Deep Neural Network, with first layer of 6 neurons for 6 features, then to understand the complexity of the dataset, various combinations of layers and neurons were tried till we arrived with a final accuracy of 87.3% with the model containing second layer of 64 neurons, then narrowing to third layer of 32 neurons, then narrowing to fourth layer of 16 neurons, then to fifth layer of 8 neurons, then to sixth layer of 4 neurons, then to seventh layer of 2 neurons and then finally since it was a classification problem, the last layer had only 1 neuron. At every step Batch normalisation were used. The activation function used is ReLU.
