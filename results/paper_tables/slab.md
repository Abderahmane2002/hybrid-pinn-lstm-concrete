| Method | $R^2$ (withheld) | RMSE (withheld) | RMSE (training) | Solver check $R^2$ |
|:---------------------------|:--------------:|:--------------:|:----------:|:----------:|
| PENN (differentiable solver) | 0.9935 ± 0.0001 | 0.0133 ± 0.0001 | 0.0125 | 0.9998 |
| Classical + regression | 0.9938 | 0.0131 | – | exact |
| PINN | 0.9826 ± 0.0071 | 0.0216 ± 0.0043 | 0.0033 | 0.860 |
| PINN without PDE residual | 0.9411 ± 0.0640 | 0.0365 ± 0.0210 | 0.0031 | – |
| Data-driven network | 0.9852 ± 0.0012 | 0.0202 ± 0.0008 | 0.0059 | – |
