# Continuous CNN For Nonuniform Time Series (ICASSP 21')

This repo provides the code to reproduce the experiments in the papers

>Hui Shi, Yang Zhang, Hao Wu, Shiyu Chang, Kaizhi Qian, Mark Hasegawa-Johnson, Jishen Zhao <cite> Continuous Convolutional Neural Network for Nonuniform Time Series <cite>

[Full Paper](https://openreview.net/pdf?id=r1e4MkSFDr), [ICSAAP Short Paper](empty)

## Dependency

The code is implemented in Python 2 with tensorflow 1.x (many years ago). We tested
the code under:

- python=2.7
- tensorflow=1.14


## Run the code

### For auto-regression tasks, run 

> python value_pred_main.py model_name exp_name config

model_name can be "CNN", "CNT" (alias to CNNT), "CCNN", "RNN", "ICNN". 

exp_name can be "sine", "glass", or "lorenz".
 
plase find the corresponding configuration file in configs/autoregression. 

For example, a complete command can be 

> python value_pred_main.py CCNN sine configs/autoregression/ccnn.json

### For point process tasks, run

> python point_process_main.py configs/bo-pp/ccnn.json

Dataset name and training split are specified in the config files. 
