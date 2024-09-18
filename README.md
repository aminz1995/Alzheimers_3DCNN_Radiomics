# Automated Classification of Alzheimer's Disease Using 3D CNNs and Radiomic Features from T1-weighted MRI
This repository contains the code for the article titled:

**"Automated Classification of Alzheimer's Disease, Mild Cognitive Impairment, and Cognitively Normal Patients Using 3D Convolutional Neural Network and Radiomic Features from T1-Weighted Brain MRI: A Comparative Study on Detection Accuracy"**

## Citation

If you use this code in your research, please cite our paper:

Amin Zarei, Ahmad Keshavarz, Esmail Jafari, Reza Nemati, Akram Farhadi, Ali Gholamrezanezhad, Habib Rostami, Majid Assadi,
Automated classification of Alzheimer's disease, mild cognitive impairment, and cognitively normal patients using 3D convolutional neural network and radiomic features from T1-weighted brain MRI: A comparative study on detection accuracy,
Clinical Imaging,
2024,
110301,
ISSN 0899-7071,
https://doi.org/10.1016/j.clinimag.2024.110301.
(https://www.sciencedirect.com/science/article/pii/S0899707124002316)

## Overview

This repository includes code for implementing a 3D Convolutional Neural Network (CNN) for the classification of Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and cognitively normal patients. The approach uses radiomic features extracted from T1-weighted brain MRI scans.

## Dependencies

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [ADNI Dataset](http://adni.loni.usc.edu/)

## Model Architecture

Below is an image of the model architecture used in this study:

![Model Architecture](images/model.png)

## Usage

1. **Clone the Repository:**
   
   ```bash
   git clone https://github.com/your_username/Alzheimers_3DCNN_Radiomics.git
   ```

2. **Install Dependencies:**
   
   ```bash
   pip install -r requirements.txt
   ```
  
3. **Run the Scripts:**

   ```bash
   python training.py
   ```

## Data
Please note that the data used for training and evaluation is not included in this repository. For access to the data, refer to the ADNI Dataset and follow their data access procedures.

## License
This project is licensed under the Apache License 2.0.

## Contact
For questions or further information, please contact [aminz1995@gmail.com].
   
