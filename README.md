# Hyperparameter Optimization for Non-linear Machine Learning Models Within the DOMINO Framework 

## Overview

### The DOMINO Framework

- **D**ata-driven **O**ptimization of bilevel **M**ixed-**I**nteger **NO**n-linear problems <sup>[1](DOMINO)<sup>
- This framework approximates the bilevel optimization problem as a single level task and solves it using data-driven optimization algorithms
- Because of the flexibility of the framework, a wide range of data-driven algorithms can be integerated to the framework
- Hyperparameter optimization for ML models (with cross-validation reaining), is formulized as a bilevel optimization problem and can by solved within the DOMINO framework
- An example of bilevel formulation for hyperparameter optimization of a ridge-regularized regression model with cross-validation training:

$$
\begin{aligned}
\min_{\lambda} \quad & \frac{1}{|\boldsymbol{K}|} \sum_{k=1}^{|\boldsymbol{K}|} \frac{1}{N^{val}_{k}} || (\boldsymbol{w^{T}_{k}}\boldsymbol{X})_{k}-\boldsymbol{y}^{val}_{k} ||^{2}_{2} \\
\text{s.t.} \quad & \min_{\boldsymbol{w_{k}}} \frac{1}{N^{trn}_{k}} || (\boldsymbol{w^{T}_{k}}\boldsymbol{X})_{k}-\boldsymbol{y}^{trn}_{k} ||^{2}_{2} + \lambda ||\boldsymbol{w_{k}}||_2^{2} \quad \forall k \in \boldsymbol{K}\\
& \boldsymbol{w_{k}} \in \mathbb{R}^m \\
& \lambda \geq 0
\end{aligned}
$$

  


## Repository Structure

```
├── main.py
├── ETOH.csv
└── 
```

## Python Libraries

- nlopt
- scikit-learn (sklearn)

## Data and Script Explanation

- Data: In this case study, We examine the separation of ethanol and water by extractive distillation using monoethylene glycol as the solvent. The procedure involves two columns arranged in series: the solvent and an azeotropic mixture of ethanol and water are simultaneously introduced into the first column to extract ethanol, and the bottom product is subsequently directed to the second column to recover and recycle the monoethylene glycol solvent. The specifics of the process design, the Aspen Plus model, and the precise entry locations of the streams to the columns are available in <sup>[2](ghalavand2021heat)</sup>. We gather 2,500 random samples within the operational range of the process variables defined in the table from the Aspen Plus simulation.  

### Table: Features of used as variables in the regression model for this case study

| Process Variable | Range |
|------------------|-------|
| Temperature (°C) (S-101) | 30 - 60 |
| Ethanol wt% (S-100) | 80 - 93 |
| Reflux ratio (T-1) | 0.3 - 0.5 |
| Reflux ratio (T-2) | 0.3 - 0.5 |
- The dataset is first standardized to a mean of zero and standard deviation of 1, and 70\% is randomly split for training-validation set with the remaining 30\% as testing set for model evaluation.
- A fourth-degree polynomial serves as the regression model for this learning problem.   Model training uses ridge regression with 5-fold cross-validation to minimize the penalized residual sum of squares loss function. The penalty weight, $\lambda$, is a model hyperparameter that will be optimized using the suggested methodology.
- For this example, the DIRECT <sup>[3](DIRECT)</sup>, from nlopt library <sup>[4](nlopt)</sup> algorithm is used within the DOMINO framework to tune the model 

## Citation

**Paper:**



**BibTeX:**

```bibtex
@article{,
  author    = {},
  title     = {},
  journal   = {},
  year      = {},
  volume    = {},
  pages     = {},
  doi       = {}
}
```
## References
<a name="references"></a>
1. Beykal, Burcu, Styliani Avraamidou, Ioannis PE Pistikopoulos, Melis Onel, and Efstratios N. Pistikopoulos. "Domino: Data-driven optimization of bi-level mixed-integer nonlinear problems." Journal of Global Optimization 78, no. 1 (2020): 1-36.
2. Ghalavand, Younes, Hasan Nikkhah, and Ali Nikkhah. "Heat pump assisted divided wall column for ethanol azeotropic purification." Journal of the Taiwan Institute of Chemical Engineers 123 (2021): 206-218.
3. Jones, Donald R., Cary D. Perttunen, and Bruce E. Stuckman. "Lipschitzian optimization without the Lipschitz constant." Journal of optimization Theory and Applications 79, no. 1 (1993): 157-181.
4. Johnson, S. G., J. D. Joannopoulos, and M. Soljačić. Nlopt library. 2007.
