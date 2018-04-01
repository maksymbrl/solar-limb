# solar-limb

> This work is based on the following project: https://www.overleaf.com/read/ywczhyybtfgs. The main idea was to write a self-consisten GUI application which will find the effective temperature of the Sun through analyzing images taken in the different wave-bands.

## Table of Contents

- [Theory](#theory)
- [Data analysis](#data-analysis)
- [Program](#program)
- [Examples](#examples)

## Theory

> Here, I briefly state main concepts and equations used. For more complete description please refer to the [link]

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

Assuming Eddington approximation with the local thermodynamic equilibrium, from \rf{15} we get

![equation](http://latex.codecogs.com/gif.latex?T^4(\tau)=\frac{3}{4}T_{eff}^4\left(\tau&plus;\frac{2}{3}\right),&space;\quad&space;\quad&space;\quad&space;(4))

Since photosphere is defined as the layer on which effective temperature $T_{eff}$ is equivalent to real temperature $T$, from \rf{4} we get

![equation](http://latex.codecogs.com/gif.latex?T_{eff}^{phot}=T\left(\tau=2/3\right),&space;\quad&space;\quad&space;\quad&space;(5))

Starting from the Planck's law expressed in terms of wavelength $\lambda$

![equation](http://latex.codecogs.com/gif.latex?B_\lambda(\lambda,&space;T)=\frac{2hc^2}{\lambda^5}&space;\frac{1}{e^{\frac{hc}{\lambda&space;k_B&space;T}}&space;-1},&space;\quad&space;\quad&space;\quad&space;(6))

we get

![equation](http://latex.codecogs.com/gif.latex?T(\tau_\lambda)=\frac{hc}{k\lambda}\frac{1}{\log\left(1&plus;\frac{2hc^2}{\lambda^5S_\lambda(\tau_\lambda)}\right)}&space;,&space;\quad&space;\quad&space;\quad&space;(7))

## Data analysis

To describe the distance from the center of the Sun, we use the distance parameter $\mu$:

\be{30}
\mu=\cos\theta=\sqrt{1-\left(\frac{r}{R}\right)^2}
\ee
where $\theta$ is the heliocentric angle, $R$ is the Sun's radius and $r$ is the current distance to the center of the disk. It is seen that distance parameter decreases with an increasing radius (being maximum in the center). 

For each single snapshot, we divide the Sun for 9 concentric rings with the center in the solar center. Every region has $N\approx 100$ points. We number the regions with the symbol $k$. The $k$-th region has a number $N_k$ of pixels and extends from the minimum distance (from the center) $r_k^{in}$ to the maximum $r_k^{out}$. The mean distance $r_k$ from the center and its associated error $\delta r_k$ (similar to the average and the semi dispersion on $r_k^{in}$ and $r_k^{out}$) are defined as:

\be{31}
r_k=\frac{r_k^{in}+r_k^{out}}{2}, \quad \delta r_k=\frac{r_k^{in}-r_k^{out}}{2}\, .
\ee

To get more information from our data, we firstly find an intensity of each pixel $I_k^n$ in the $k$-th region and then get an average intensity $I_k$ in this ring. as error we take the standard deviation over the $I_k^n$. We also define the distance parameter $\mu_k$ of the $k$-th region 

\be{32}
I_k=\frac{1}{N_k}\sum_{n=1}^{N_k}I_k^n\pm\Delta_k I_k, \quad \mu_k=\sqrt{1-\left(\frac{r_k}{R}\right)^2}\pm\frac{r_k}{\sqrt{1-\left(\frac{r_k}{R}\right)^2}}\delta r_k\, .
\ee

Note that $\delta\mu_k$ is not a measure error, but depends on the way in which we define the regions on the diameter.

~
During our observations we have collected 20 snapshots of the Sun for 4 positions of the filter wheel (see details in section \rf{sec:instrumentation}). In order to analyze them, we developed an algorithm in the Python environment. The program goes filter by filter, repeated over all 20 snapshots (collected in our experiment) and finds the coordinates of the center, as well as four recognizable points as (Left, Right, Top and Bottom of the solar snapshot, from the point of view of our CCD camera, see left figures \rf{fig:filters}), defines the regions on the diameters and calculates all the quantities from above. At last, the final intensity $I_k^{FIN}$ and the distance parameters $\mu_k^{FIN}$ are obtained (for the given filter) through the weighted average over all twenty images.

Referring to the central intensity $I_\lambda\left(0, 1\right)$ and restricting ourselves to a second order polynomial fit, we can write

\be{33}
\frac{I_\lambda(0,\mu)}{I_\lambda(0,1)}=a_0+a_1\mu+2a_2\mu^2, \quad \frac{S_\lambda(\tau_\lambda)}{I_\lambda(0,1)}=a_0+a_1\tau+a_2\tau_\lambda^2\, ,
\ee

~
In order to avoid non-physical behaviors, we limit ourselves with the second order expansion (as it was done in \cite{Zuliani}). We are interested in relative intensities $I_\lambda(0, \mu)/I_\lambda(0, 1)$ and the least square polynomial fit allows us to find the fit coefficients ($a_0$, $a_1$ and $a_2$).

~
We plot the relative intensities $I_\lambda(0, \mu)/I_\lambda(0, 1)$ in the figure \rf{fig:int_miu} referring with the different symbols to the various filters' measures.

In the whole discussion we refer to relative intensities. In particular all the intensity values we treat are expressed in digital units and we usually normalize all the values relatively to the center of the disk. But, to obtain the source function we need physical central intensity $I_{\lambda}(0, 1)$ expressed in $Wm^{-3}sr^{-1}$ units. Following the reasoning of \cite{Zuliani}, we could not measure them, so we can use a set of tabulated intensities determined for some specific wavelengths (see table \ref{table:interpolatedIntensity}). 

|Band | Wavelength (nm) | $I_{\lambda}(0,1)$ ($Wm^{-3}sr^{-1}$) | 
| :---: | :---: | :---: |
| B | 420 | $(4,5 \pm 0,6) \times 10^{13}$ |
| V | 547 | $(3,6 \pm 0,2) \times 10^{13}$ |
| R | 648 | $(2,8 \pm 0,3) \times 10^{13}$ |
| I | 871 | $(1,6 \pm 0,5) \times 10^{13}$ |

Then we extrapolated the temperatures for $\tau=2/3$ for each filter and found a values for the effective temperatures: we list them in the table \ref{table:Teff}. We applied the uncertainty propagation theory for calculation of the associated errors on the wavelength errors $\delta\lambda=FWHM/2$ (the FWHM of the filter transparency profile), the central intensities $\delta I_\lambda$ and the fit coefficients $\delta a_i$ ($i=0,1,2$).

In the end, we used a weighted mean for B,V and R (excluding I) band filters as
\be{34}
T_{eff}=\cfrac{\sum_i T_{eff}^{(i)}\sigma_i^{-2}}{\sum_i \sigma_i^{-2}}\, ,
\ee
together with associated uncertainties $\sigma_i$, supposing them being small, i.e., neglecting the higher orders of smallness (for example, $\sigma_i \sigma_j \ll \sigma_i$) 
\be{35}
\sigma_i=T_{eff} \left[ \sum_i\left(\cfrac{\sigma_i}{T_{eff}^{(i)}}\right)^2 \right]^{-1/2}\, ,
\ee
to obtain the final effective temperature
\be{36}
T_{eff}=T\left(\tau=\frac{2}{3}\right)\, .
\ee

## Program



## Examples
