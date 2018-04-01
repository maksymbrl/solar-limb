# solar-limb

> This work is based on the following project: https://www.overleaf.com/read/ywczhyybtfgs. The main idea was to write a self-consisten GUI application which will find the effective temperature of the Sun through analyzing images taken in the different wave-bands.

## Table of Contents

- [Theory](#theory)
- [Data analysis](#data-analysis)
- [Program](#program)
- [Examples](#examples)

## Theory

> Here, I briefly state main concepts and equations used. For more complete description please refer to the https://www.overleaf.com/read/ywczhyybtfgs

In general, to solve radiative trasfer equation

![equation](http://latex.codecogs.com/gif.latex?\frac{dI_\lambda}{\kappa_\lambda&space;ds}=\frac{dI_\lambda}{d\tau_{\lambda&space;s}}=-I_\lambda&space;&plus;&space;\frac{\epsilon_\lambda}{\kappa_\lambda}=-I_\lambda&space;&plus;&space;S_\lambda,&space;\quad&space;\quad&space;\quad&space;(1))


is tedious task. Luckily, some simplifications can be made in study of stellar atmospheres:

- Stellar atmosphere can usually be well approximated as a plane-parallel system -> Plane-parallel approximation;
- We assume the emission being thermal;
- We assume that the atmosphere is homogeneous and isotropic;

Thus, we have:

![eqution](http://latex.codecogs.com/gif.latex?I^{(1)}_\lambda&space;=&space;B_\lambda&space;-\frac{\cos\theta}{\kappa_\lambda}\frac{\partial&space;B_\lambda}{\partial&space;z}&space;,&space;\quad&space;\quad&space;\quad&space;(2))

and the source function is the product of power series:

![equation](http://latex.codecogs.com/gif.latex?S_\lambda=I_\lambda\left(0,1\right)\sum_{n=0}^m&space;a_{\lambda&space;n}\tau_\lambda^n,&space;\quad&space;I_\lambda(0,&space;\mu)=I_\lambda(0,&space;1)=\sum_{n=0}^m&space;a_{\lambda&space;n}n!\mu^n,&space;\quad&space;\quad&space;\quad&space;(3))

Assuming Eddington approximation with the local thermodynamic equilibrium, we get

![equation](http://latex.codecogs.com/gif.latex?T^4(\tau)=\frac{3}{4}T_{eff}^4\left(\tau&plus;\frac{2}{3}\right),&space;\quad&space;\quad&space;\quad&space;(4))

Since photosphere is defined as the layer on which effective temperature ![equation](http://latex.codecogs.com/gif.latex?T_{eff}) is equivalent to real temperature T, we get

![equation](http://latex.codecogs.com/gif.latex?T_{eff}^{phot}=T\left(\tau=2/3\right),&space;\quad&space;\quad&space;\quad&space;(5))

Starting from the Planck's law expressed in terms of wavelength ![equation](http://latex.codecogs.com/gif.latex?\lambda)

![equation](http://latex.codecogs.com/gif.latex?B_\lambda(\lambda,&space;T)=\frac{2hc^2}{\lambda^5}&space;\frac{1}{e^{\frac{hc}{\lambda&space;k_B&space;T}}&space;-1},&space;\quad&space;\quad&space;\quad&space;(6))

we get

![equation](http://latex.codecogs.com/gif.latex?T(\tau_\lambda)=\frac{hc}{k\lambda}\frac{1}{\log\left(1&plus;\frac{2hc^2}{\lambda^5S_\lambda(\tau_\lambda)}\right)}&space;,&space;\quad&space;\quad&space;\quad&space;(7))

## Data analysis

- We divide the Sun for 9 concentric rings with the center in the solar center and number each region with symbol k (each region has ![equation](http://latex.codecogs.com/gif.latex?N\approx&space;100) points). 
- The  k-th region extends from the minimum distance (from the center) ![equation](http://latex.codecogs.com/gif.latex?r_k^{in}) to the maximum ![equation](http://latex.codecogs.com/gif.latex?r_k^{out}). 
- The mean distance ![equation](http://latex.codecogs.com/gif.latex?r_k) from the center and its associated error ![equation](http://latex.codecogs.com/gif.latex?\delta&space;r_k) are defined as:

![equation](http://latex.codecogs.com/gif.latex?r_k=\frac{r_k^{in}&plus;r_k^{out}}{2},&space;\quad&space;\delta&space;r_k=\frac{r_k^{in}-r_k^{out}}{2}&space;,&space;\quad&space;\quad&space;\quad&space;(8))

- We firstly find an intensity of each pixel ![equation](http://latex.codecogs.com/gif.latex?I_k^n) in the k-th region and then get an average intensity ![equation](http://latex.codecogs.com/gif.latex?I_k) in this ring. As error we take the standard deviation over the ![equation](http://latex.codecogs.com/gif.latex?I_k^n). We also define the distance parameter ![equation](http://latex.codecogs.com/gif.latex?\mu_k) of the k-th region 

![equation](http://latex.codecogs.com/gif.latex?I_k=\frac{1}{N_k}\sum_{n=1}^{N_k}I_k^n\pm\Delta_k&space;I_k,&space;\quad&space;\mu_k=\sqrt{1-\left(\frac{r_k}{R}\right)^2}\pm\frac{r_k}{\sqrt{1-\left(\frac{r_k}{R}\right)^2}}\delta&space;r_k,&space;\quad&space;\quad&space;\quad&space;(9))

Note that ![equation](http://latex.codecogs.com/gif.latex?\delta\mu_k) is not a measure error, but depends on the way in which we define the regions on the diameter.

- Referring to the central intensity ![equation](http://latex.codecogs.com/gif.latex?I_\lambda\left(0,&space;1\right)) and restricting ourselves to a second order polynomial fit, we can write

![equation](http://latex.codecogs.com/gif.latex?\frac{I_\lambda(0,\mu)}{I_\lambda(0,1)}=a_0&plus;a_1\mu&plus;2a_2\mu^2,&space;\quad&space;\frac{S_\lambda(\tau_\lambda)}{I_\lambda(0,1)}=a_0&plus;a_1\tau&plus;a_2\tau_\lambda^2,&space;\quad&space;\quad&space;\quad&space;(10))

- In order to avoid non-physical behaviors, we limit ourselves with the second order expansion. We are interested in relative intensities ![equation](http://latex.codecogs.com/gif.latex?I_\lambda(0,&space;\mu)/I_\lambda(0,&space;1)) and the least square polynomial fit allows us to find the fit coefficients (![equation](http://latex.codecogs.com/gif.latex?a_0,&space;\quad&space;a_1&space;\quad&space;and&space;\quad&space;a_2)).

In the whole discussion we refer to relative intensities. In particular all the intensity values we treat are expressed in digital units and we usually normalize all the values relatively to the center of the disk. But, to obtain the source function we need physical central intensity ![equation](http://latex.codecogs.com/gif.latex?I_{\lambda}(0,&space;1)) expressed in ![equation](http://latex.codecogs.com/gif.latex?Wm^{-3}sr^{-1}) units.

|Band | Wavelength (nm) | ![equation](http://latex.codecogs.com/gif.latex?I_{\lambda}(0,1)) (![equation](http://latex.codecogs.com/gif.latex?Wm^{-3}sr^{-1})) | 
| :---: | :---: | :---: |
| B | 420 | ![equation](http://latex.codecogs.com/gif.latex?(4,5&space;\pm&space;0,6)&space;\times&space;10^{13}) |
| V | 547 | ![equation](http://latex.codecogs.com/gif.latex?(3,6&space;\pm&space;0,2)&space;\times&space;10^{13}) |
| R | 648 | ![equation](http://latex.codecogs.com/gif.latex?(2,8&space;\pm&space;0,3)&space;\times&space;10^{13}) |
| I | 871 | ![equation](http://latex.codecogs.com/gif.latex?(1,6&space;\pm&space;0,5)&space;\times&space;10^{13}) |

- Then we extrapolated the temperatures for ![equation](http://latex.codecogs.com/gif.latex?\tau=2/3) for each filter and found a values for the effective temperatures. 
- We applied the uncertainty propagation theory for calculation of the associated errors on the wavelength errors ![equation](http://latex.codecogs.com/gif.latex?\delta\lambda=FWHM/2) (the FWHM of the filter transparency profile), the central intensities ![equation](http://latex.codecogs.com/gif.latex?\delta&space;I_\lambda) and the fit coefficients ![equation](http://latex.codecogs.com/gif.latex?\delta&space;a_i,&space;\quad&space;(i=0,1,2)).
- In the end, we used a weighted mean for various band filters as

![equation](http://latex.codecogs.com/gif.latex?T_{eff}=\cfrac{\sum_i&space;T_{eff}^{(i)}\sigma_i^{-2}}{\sum_i&space;\sigma_i^{-2}},&space;\quad&space;\quad&space;\quad&space;(11))

together with associated uncertainties ![equation](http://latex.codecogs.com/gif.latex?\sigma_i), supposing them being small, i.e., neglecting the higher orders of smallness

![equation](http://latex.codecogs.com/gif.latex?\sigma_i=T_{eff}&space;\left[&space;\sum_i\left(\cfrac{\sigma_i}{T_{eff}^{(i)}}\right)^2&space;\right]^{-1/2},&space;\quad&space;\quad&space;\quad&space;(12))

to obtain the final effective temperature

![equation](http://latex.codecogs.com/gif.latex?T_{eff}=T\left(\tau=\frac{2}{3}\right),&space;\quad&space;\quad&space;\quad&space;(13))

## Program



## Examples
