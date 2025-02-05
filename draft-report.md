---
title: "Decision Making Project Report"
author:
    - Clovis Piedallu
    - Basile Heurtault
    - Jérémy Mathet
date: \today
# abstract: |
#     Brief summary of your report's main points and findings.
#     (Replace this with your actual abstract)
# documentclass: report
toc: true
numbersections: false
# bibliography: references.bib
link-citations: true
geometry:
    - margin=2.5cm
header-includes:
    - \usepackage{graphicx}
    - \usepackage{amsmath}
    - \usepackage{hyperref}
    - \usepackage{float}
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[L]{Decision Making Project}
    - \fancyhead[R]{\thepage}
    - \fancyfoot[C]{Piedallu, Heurtault, Mathet}
---

# Introduction

# Problem formulation


## Objective function

We seek to minimize :

$$
\min \sum_{k=1}^{K} \sum_{j=1}^{P} \sigma^k (j)
$$

## Decision variables

- $u_i^k(x_i^l) \geq 0$ : Continuous decision variable $\qquad \forall i \in [1,n], \quad \forall l \in [1,L], \quad \forall k \in [1,K]$
- $c_k^{(j)} \in \{0,1\}$ : Binary decision variable $\qquad \forall j \in [1,P], \quad \forall k \in [1,K]$


## Constraints

The problem is subject to the following constraints:

### Normalization constraints

$$
u_i^k(x_i^0) = 0 \quad \forall i \in [1,n], \forall k \in [1,K]
$$
$$
\sum_{i=1}^{n} u_i^k(x_i^L) = 1 \quad \forall k \in [1,K]
$$

### Monotonicity constraints

$$
u_i^k(x_i^{l+1})-u_i^k(x_i^{l}) \geq \epsilon \quad \forall l \in [0,L]
$$

### Preference constraints

$$
M(c_k^{(j)}-1) \leq \sum_{i=1}^{n} [u_i^k(x_i^{(j)}) - u_i^k(y_i^{(j)})] + \sigma^k (x^{(j)}) \leq Mc_k^{(j)} \quad \forall k \in [1,K], \forall j \in [1,P]
$$
$$
\sum_{k=1}^{K} c_k^{(j)} \geq 1 \quad \forall j
$$

### Domain constraints

$$
0 \leq u_i^k(x_i^l) \leq 1 \quad \forall i, \forall k, \forall l
$$
$$
c_k^{(j)} \in \{0,1\} \quad \forall j,  \forall k
$$


