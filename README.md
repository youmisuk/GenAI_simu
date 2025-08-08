# Code for *Using Generative AI for Sequential Data Generation in Monte Carlo Simulation Studies*

## 0 Table of Contents

1. [Overview](#1-overview)
2. [Running the Code](#2-running-the-code)  


## 1 Overview  

**Title**: Using Generative AI for Sequential Data Generation in Monte Carlo Simulation Studies

**Authors**: Youmi Suk, Chenguang Pan, Ke Yang

Monte Carlo simulation studies are the de facto standard for assessing both classical and modern statistical methods. In traditional simulation studies, researchers have full discretion over data generation and often create simulated data with a high degree of smoothness and limited dependencies among variables. Under such conditions, the performance of statistical methods may not indicate their real-world performance. To address this issue, we propose a novel AI-based simulation framework that leverages generative AI (GenAI) to create realistic synthetic data and incorporate them into Monte Carlo simulation studies. In particular, we focus on simulation studies with sequential data (e.g., action sequences from process data) and demonstrate our approach by evaluating the predictive performance of an example statistical method. Our framework consists of five key steps: (i) pre-processing input data, (ii) training GenAI models on input data, (iii) assessing synthetic data quality, (iv) conducting AI-based simulations, and (v) evaluating simulation results. We also perform robustness checks on both synthetic data quality and simulation outcomes by modifying specific steps of our approach. Overall, the proposed AI-based simulations outperform traditional simulations in generating realistic synthetic data and providing a more accurate evaluation of the real-world performance of statistical methods.

For more details of our proposed methods, see [our paper](https://osf.io/preprints/psyarxiv/7rd86_v2). 

## 2 Running the Code  
### 2.1 Requirements  
#### 2.1.1 Requirements for dataset formats  
The CTGAN model is trained on a single-table dataset as shown in Table 1, where the ActionSequence column is treated as categorical. In contrast, CPAR uses a sequential-table dataset as shown in Table 2, where each action is a separate row observation, and each test-taker contributes multiple rows (denoted as the Action column).

**Table 1: Single table dataset for CTGAN**
| ID       | Gender | Age | Score | ActionSequence         | ResponseTime |
|----------|--------|-----|-------|------------------------|--------------|
| 1001041  | Male   | 16  | 1     | START_COMBOBOX,...,END | 75.12        |
| 1001042  | Female | 40  | 1     | START_TAB,...,END      | 105.38       |
| 1001045  | Female | 35  | 0     | START_NEXT_INQUIRY,END | 114.77       |
| ...      | ...    | ... | ...   | ...                    | ...          |

  
**Table 2: Sequential table dataset for CPAR**  
| ID       | Gender | Age | Score | Action    | ResponseTime |
|----------|--------|-----|-------|-----------|--------------|
| 1001041  | Male   | 16  | 1     | START     | 75.12        |
| 1001041  | Male   | 16  | 1     | COMBOBOX  | 75.12        |
| 1001041  | Male   | 16  | 1     | ...       | 75.12        |
| 1001041  | Male   | 16  | 1     | END       | 75.12        |
| 1001042  | Female | 40  | 1     | START     | 105.38       |
| 1001042  | Female | 40  | 1     | TAB       | 105.38       |
| 1001042  | Female | 40  | 1     | TAB       | 105.38       |
| ...      | ...    | ... | ...   | ...       | ...          |  
  
#### 2.1.2 Requirements for computational resources 
  
A GPU is recommended for training generative AI models. For the task of synthetic tabular data, a consumer-grade gaming GPU with CUDA cores is sufficient. Alternatively, free GPU resources, like those provided by Google Colab, can be used for training GenAI models.  
We conducted all experiments on a local machine equipped with an Intel Core i7-14700F CPU and an NVIDIA GeForce RTX 4070 Super GPU. Training an optimized CTGAN model took approximately 10 minutes, while training a CPAR model required about 25 minutes.  

The following table shows the pipeline differences between CTGAN and CPAR:  

| Component               | CTGAN                  | CPAR                  |
|-------------------------|------------------------|-----------------------|
| Input data type         | single-table           | sequential-table      |
| Auxiliary variables     | action count           | action count, sequence index |
| Data encoder: ActionSequence | uniform encoder     | -                     |
| Metadata: ID            | ID                     | ID, sequence index    |
| Tunable parameters      | number of hidden layers<br>number of neurons per layer<br>number of epochs, etc | number of candidate sequences<br>number of epochs, etc |
| Training time           | 10 minutes             | 25 minutes            |
| Sample generation time  | 1 second               | 13 minutes            |




#### 2.1.3 Key libraries  

To run the Python files, please install the following Python version and libraries.  

```{python}
python==3.12.3
SDV==1.21.0
pytorch==2.3.1
```
To run the R files, please install the following R packages.

```{R}
install.packages(c(
  "doParallel",
  "foreach",
  "ProcData",
  "ggplot2",
  "patchwork"
))

```

### 2.2 Reproduce the results
1. Create the conda virtual environment first with the required versions of python and libraries. Run this project in this virtual env.
2. Run the jupyter notebook "02_code\\01_CTGAN_optimal.ipynb" to reproduce the optimal CTGAN.
3. Run the jupyter notebook "02_code\\02_CPAR.ipynb" to reproduce the CPAR.
4. Run the R file "02_code\\03_Run_Tangs_sim_using_Synthetic_data.R" and "02_code\\04_Analysze_sim_results.R" to reproduce the results for using synthetic data for Tang's simulation, i.e., Figure 5. Please [download the synthetic data](https://drive.google.com/drive/folders/1zq5JiHq77efVdER9fTUVLbJYPvPQLkNa?usp=sharing) into the 03_outputs folder before running the R file.




