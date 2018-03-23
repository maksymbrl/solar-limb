# solar-limb

> This work is based on the following project: [link]. The main idea was to write a self-consisten GUI application which will find the effective temperature of the Sun through analyzing images taken in the different wave-bands.

## Table of Contents

- [Theory](#theory)
- [Program](#program)
- [Numerical Integration](#numerical-integration)
- [Examples](#examples)



## Theory

> Here, I briefly state main concepts and equations used. For more complete description please refer to the [link]

In general, to solve radiative trasfer equation

\be{7}
\frac{dI_\lambda}{\kappa_\lambda ds}=\frac{dI_\lambda}{d\tau_{\lambda s}}=-I_\lambda + \frac{\epsilon_\lambda}{\kappa_\lambda}=-I_\lambda + S_\lambda\, ,
\ee

is tedious task. Luckily, some simplifications can be made in study of stellar atmospheres:

- Stellar atmosphere can usually be well approximated as a plane-parallel system -> Plane-parallel approximation;
- We assume the emission being thermal;
- We assume that the atmosphere is homogeneous and isotropic;

Thus, we have:

\be{14}
I^{(1)}_\lambda = B_\lambda -\frac{\cos\theta}{\kappa_\lambda}\frac{\partial B_\lambda}{\partial z} \, ,
\ee

and the source function is the product of power series:

\be{15}
S_\lambda=I_\lambda\left(0,1\right)\sum_{n=0}^m a_{\lambda n}\tau_\lambda^n, \quad I_\lambda(0, \mu)=I_\lambda(0, 1)=\sum_{n=0}^m a_{\lambda n}n!\mu^n\, ,
\ee

Assuming Eddington approximation with the local thermodynamic equilibrium, from \rf{15} we get

\be{18}
T^4(\tau)=\frac{3}{4}T_{eff}^4\left(\tau+\frac{2}{3}\right)\, .
\ee

Since photosphere is defined as the layer on which effective temperature $T_{eff}$ is equivalent to real temperature $T$, from \rf{4} we get

\be{19}
T_{eff}^{phot}=T\left(\tau=2/3\right)\, .
\ee

Starting from the Planck's law expressed in terms of wavelength $\lambda$

\be{20}
B_\lambda(\lambda, T)=\frac{2hc^2}{\lambda^5} \frac{1}{e^{\frac{hc}{\lambda k_B T}} -1} \, ,
\ee

we get

\be{22}
T(\tau_\lambda)=\frac{hc}{k\lambda}\frac{1}{\log\left(1+\frac{2hc^2}{\lambda^5S_\lambda(\tau_\lambda)}\right)} \, .
\ee


## Program
