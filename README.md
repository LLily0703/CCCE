## CCCE
UAI2023 "Conditional Counterfactual Causal Effect for Individual Attribution"

### Introduction
Identifying the causes of an event, also termed as causal attribution, is a commonly encountered task in many application problems.

Available methods, mostly in Bayesian or causal inference literature, suffer from two main drawbacks:
  - cannot attributing for individuals
  - attributing one single cause at a time and cannot deal with the interaction effect among multiple causes.
 
In this paper, based on our proposed new measurement, called conditional counterfactual causal effect (CCCE), we introduce
an individual causal attribution method, which is able to utilize the individual observation as the evidence and consider common influence and interaction effect of multiple causes simultaneously. We discuss the identifiability of CCCE and also give the identification formulas under proper assumptions. 

### Experiment
We conduct experiments on simulated and real data to illustrate the effectiveness of CCCE and the results show that our proposed method outperforms significantly over state-of-the-art methods.

#### Requirement
igraph==0.10.4  
networkx==3.1  
numpy==1.24.3  
pandas==2.0.2  
python-dateutil==2.8.2  
pytz==2023.3  
six==1.16.0  
texttable==1.6.7  
tqdm==4.65.0  
tzdata==2023.3  

#### Simulated data setting
First, we have provided the model that we trained for simply reproducing the results in our paper. Then you can run the following commands to reproduce the results in our paper.  
```
python -u main.py --random_num 3000 --sample_num 1000 --save_path './' 
```
If you want to test the effect of synergy in a simulation scenario, you can uncomment the following code in the 'simulate' function of main.py
```
# for interaction_effect simulation
C = B * (np.random.rand(B.shape[0], B.shape[1]) * 0.5)  # [0,0.5]
C[0, 3] = np.random.rand() * 0.4 + 0.6  # [0.6,1]
W = C  # weights for each edge
co_list = [2, 4]
weight = 2
node_beta = np.random.rand(B.shape[0]) * 3 - 2  # [-2,2]
node_beta[0] = np.random.rand() + 1 
counter_dict = generate_counter_dict_interaction_effect(W, node_beta, buy_index, co_list, weight)
```
For real data, you can run the 'real' function in main.py to get the result.

