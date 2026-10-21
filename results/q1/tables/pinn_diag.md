| Run | Total loss | Data term | PDE term | Training RMSE | PDE residual RMS | Solver check $R^2$ | Bias of log $K_{\text{int}}$ |
|:------------------------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| As in the comparison | 0.135 | 0.113 | 0.0021 | 0.0034 | 0.048 | 0.864 | -1.13 |
| Parameters started at classical values | 0.137 | 0.113 | 0.0024 | 0.0034 | 0.049 | 0.893 | -1.08 |
| PDE weight λ = 100 | 0.186 | 0.107 | 0.0007 | 0.0033 | 0.034 | 0.898 | -1.03 |
| Synthetic training data (known parameters) | 0.020 | 0.001 | 0.0019 | 0.0003 | 0.046 | 0.828 | -1.13 |
| Parameters fixed at classical values | 0.135 | 0.101 | 0.0033 | 0.0032 | 0.107 | 0.999 | +0.00 |
