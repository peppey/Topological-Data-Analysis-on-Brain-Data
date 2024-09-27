# Topological Data Analysis on Multimodal Brain Data


Code for the master thesis "Topological Data Analysis on Multimodal Brain Data" by Pia Baronetzky (supervisors Bastian Rieck and Silviu Bodea).



## Instructions

This code is tested on a Mac with Python version 3.10.12.

For installing the packages and dependencies, run

`poetry install`


The `Anesthesia_Data` folder contains code for classifying multimodal brain data into anesthesia stages.
Within the `Anesthesia_Data` folder, the `Time_Series` folder contains code for creating machine learning features from the EEG/EMG part of the data. The `Brain_Imaging` folder contains code for creating features from the brain imaging part of the data.


The `Sleep_Data` folder contains code for classifying EEG/EMG data into sleep stages.

`Utils` contains helpers for both data types.


Both the `Anesthesia_Data` and `Sleep_Data`folders require a folder `Data` containing the data, as well as folders `Embeddings_and_Persistence_Diagrams`, `Features`, `Plots` and `Train_Test_Splitting`. Some of these folders require subfolders for the single subjects, but this should become clear from the code.


In both folders `Anesthesia_Data` and `Sleep_Data, there is a file `Classification.ipynb` for the main classification (and additionally, a file `Classification_Statistical_Features.ipynb` for the baseline classification methods using statistical features). Thes classification files depend on features that first have to be generated.

In order to generate the features, run the files `Preprocessing_And_Computing_Persistence_Diagrams.ipynb` first. Then, run all files ending with `_Features.ipynb`. You can then finally run the `Classification.ipynb` files.


The `Data_Exploration.ipynb` require the `Compute_Wasserstein_Barycenters.ipynb` to be run first.




