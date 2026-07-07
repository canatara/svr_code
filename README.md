# Statistical Mechanics of Support Vector Regression

Code to reproduce the figures in:

> **Statistical Mechanics of Support Vector Regression**
> Abdulkadir Canatar and SueYeon Chung, *Phys. Rev. E* **112**, 025301 (2025).
> [Journal](https://journals.aps.org/pre/abstract/10.1103/78dr-c4xd) · [arXiv:2412.05439](https://arxiv.org/abs/2412.05439) · [Physics Viewpoint](https://physics.aps.org/articles/v19/26)

## About

The paper develops a statistical-mechanics (mean-field / replica) theory of **linear ε-insensitive Support Vector Regression (SVR)**, motivated by continuous decoding from neural representations. Solving self-consistent saddle-point equations for the free energy yields closed-form predictions for the **training and generalization error learning curves** as a function of load α = P/N. Key results reproduced here:

- The ε-insensitive tube induces an **effective sample size** α_eff = α·erfc(ε/√2) (only samples outside the tube contribute).
- A **phase transition** in the training error at a critical load, and a **double-descent** generalization curve in which the tolerance ε acts as a regularizer.
- **Optimal** hyperparameters ε_opt and ridge λ_opt derived from the theory.
- Application to **neural object manifolds** (gratings, sticks, rotated MNIST digits) represented by random vs. trained ResNet features, linking representation geometry (effective/task-aligned dimension, error-mode radius) to decoding precision.

## Citation

```bibtex
@article{canatar2025svr,
  title   = {Statistical Mechanics of Support Vector Regression},
  author  = {Canatar, Abdulkadir and Chung, SueYeon},
  journal = {Physical Review E},
  volume  = {112},
  pages   = {025301},
  year    = {2025},
  doi     = {10.1103/78dr-c4xd},
  note    = {arXiv:2412.05439}
}
```
