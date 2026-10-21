| Method | $R^2$ | RMSE | Training time |
|:------------------------------------|:--------------:|:--------------:|:-----------:|
| Two-stage, linear (OLS) | 0.9934 | 0.0135 | 555 s^b^ |
| Two-stage, ridge | 0.9933 | 0.0136 | 555 s^b^ |
| Two-stage, quadratic | 0.9939 | 0.0130 | 555 s^b^ |
| Two-stage, Gaussian process | 0.9849 | 0.0204 | 555 s^b^ |
| PENN | 0.9931 ± 0.0002 | 0.0137 ± 0.0002 | 1 s^a^ |
| PENN, revised (no weight decay) | 0.9937 ± 0.0000 | 0.0132 ± 0.0000 | 0 s^a^ |
| PENN, linear composition map | 0.9928 ± 0.0001 | 0.0141 ± 0.0001 | 1 s^a^ |
| PENN, 1 × 8 units | 0.9932 ± 0.0000 | 0.0137 ± 0.0000 | 1 s^a^ |
| Data-driven network | 0.9852 ± 0.0012 | 0.0202 ± 0.0008 | 12 s |
| *Reference: classical fit to each withheld mix* | *0.9986* | *0.0063* | *41 s per mix* |
