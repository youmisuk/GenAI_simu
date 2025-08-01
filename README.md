# Code for *Using Generative AI for Sequential Data Generation in Monte Carlo Simulation Studies*

## 0 Table of Contents

1. [Overview](#1-overview)
2. [Running the Code](#2-running-the-code)  


## 1 Overview  

**Title**: Using Generative AI for Sequential Data Generation in Monte Carlo Simulation Studies

**Authors**: Youmi Suk, Chenguang Pan, Ke Yang

Monte Carlo simulation studies are the de facto standard for assessing both classical and modern statistical methods. In traditional simulation studies, researchers have full discretion over data generation and often create simulated data with a high degree of smoothness and limited dependencies among variables. Under such conditions, the performance of statistical methods may not indicate their real-world performance. To address this issue, we propose a novel AI-based simulation framework that leverages generative AI (GenAI) to create realistic synthetic data and incorporate them into Monte Carlo simulation studies. In particular, we focus on simulation studies with sequential data (e.g., action sequences from process data) and demonstrate our approach by evaluating the predictive performance of an example statistical method. Our framework consists of five key steps: (i) pre-processing input data, (ii) training GenAI models on input data, (iii) assessing synthetic data quality, (iv) conducting AI-based simulations, and (v) evaluating simulation results. We also perform robustness checks on both synthetic data quality and simulation outcomes by modifying specific steps of our approach. Overall, the proposed AI-based simulations outperform traditional simulations in generating realistic synthetic data and providing a more accurate evaluation of the real-world performance of statistical methods.

For more details of our proposed methods, see [our paper](https://osf.io/preprints/psyarxiv/7rd86_v1). 

## 2 Running the Code  
### 2.1 Requirments
```{python}
python==3.12.3
SDV==1.21.0
pytorch==2.3.1
```

### 2.2 Reproduce the results
1. Create the conda virtual environment first with the required versions of python and libraries. Run this project in this virtual env.
2. Run the jupyter notebook "02_code\\01_CTGAN_optimal.ipynb" to reproduce the optimal CTGAN.
3. Run the jupyter notebook "02_code\\02_CPAR.ipynb" to reproduce the CPAR.
4. 




