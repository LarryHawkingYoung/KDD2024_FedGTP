# KDD2024_FedGTP
Official PyTorch implementation of "FedGTP: Exploiting Inter-Client Spatial Dependency in Federated Graph-based Traffic Prediction".

## Setup
### Environment
PyTorch 1.13.1 and conducted on Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz and four NVIDIA A100-PCIE-40GB GPUs with CUDA 11.6.
```bash
conda create -n fedgtp "python=3.11"
conda activate fedgtp
bash install.sh
```

### Run
```bash
bash run.sh
```

## Default Hyperparameter Configuration
| Hyperparameters               | Values |
|-------------------------------|--------|
| hidden feature dimension $F$        | 64     |
| number of hidden layers $L$         | 2      |
| embedding dimension $d$             | 2      |
| polynomial coefficient $K$          | 4      |
| learning rate $\eta$                | 0.003  |
| batch size                              | 64     |
| number of global epochs $R_g$       | 200    |
| number of local epochs $R_l$        | 2      |
| validation ratio                        | 0.2    |
| testing ratio                           | 0.2    |
