---
layout:     post
title:      "Disentangling Disentanglement: How VAEs Learn Independent Components"
date:       2024-12-15 21:39:00 +0100
author:     Carl Allen
paper-link: https://arxiv.org/pdf/2410.22559
link-text:  "[arXiv]"
categories: Theory

---
{% include_relative _includes/head.html %}
.image1 {
  display: block;
  margin: 0 auto
}

figcaption {
  text-align: center;
}

**Disentanglement** is an intriguing phenomenon in  machine learning observed in generative models, particularly Variational Autoencoders (VAEs). Disentanglement is not a rigorously defined term but refers to when semantically meaningful factors of the data map to distinct dimensions in the latent space. This allows, for example, face images to be generated that vary only in orientation or hair colour by changing a *single latent dimension*.

<figure>
        <img src="/assets/disentanglement/faces.png" 
                alt="Faces" 
                width="420" 
                height="130" 
                class="image1" />
        <figcaption>Figs from $\beta$-VAE (Higgins et al, 2017)</figcaption>
</figure>

**Motivation**: Disentanglement is particularly intriguing because VAEs were not designed to achieve it, thus its understanding may provide new insight into how and what VAEs learn. More generally, the ability to separate independent aspects of the data could be useful in many machine learning domains and, by teasing apart their generative factors, potentially offer fundamental insights into the underlying data itself. Understanding disentanglement is of further interest since it occurs in settings (e.g. spherically symmetric prior) in which disentanglement has been considered impossible.

<img src="/assets/disentanglement/chairs.png" 
        alt="Chairs" 
        class="image1" />


Several recent works suggest that disentanglement in VAEs may stem from commonly used *diagonal posterior covariance matrices* promoting *column-orthgonality in the decoder's Jacobian matrix*. In this post, we summarise [Allen (2024)][paper], which (A) clarifies this connection, and (B) explains how it leads to disentanglement, showing that disentanglement equates to factorising the data distribution into *statistically independent components*. 
$$
\begin{equation}
\text{diag. posterior covariance} 
    \ \  \overset{A}{\Rightarrow}\ \   \text{column-orthog. Jacobian} 
    \ \ \overset{B}{\Rightarrow}\ \    \text{disentanglement}
\end{equation}
$$

**Notation**: $$\quad$$ data $$x\in\mathcal{X} \subseteq\mathbb{R}^n,\quad \text{ latent variables }z\in\mathcal{Z} =\mathbb{R}^m.$$

---


## High-level Summary: 
* The linear case $$x=Dz\ \ (D\in\mathbb{R}^{n\times m})$$ corresponds to probabilistic PCA with known analystic solutions, but a VAE with diagonal posterior covariance finds a specific subset of those solutions where latent dimensions $$z_i$$ *map to independent factors* of variation in the Gaussian data distribution.
* Surprisingly, this extends analogously to non-linear VAEs where diagonal posterior covariances encourage columns of the decoder's Jacobian to be orthogonal, causing independent latent variables to pass through the decoder separably and emerge in $$\mathcal{X}$$ over statistically independent sub-manifolds on the decoder-defined manifold. 
* Since the VAE's objective is maximised when the model distribution matches that of the data, if the data distribution has statistically independent factors, then statistically independent curves of the decoder-defined manifold align with those of the data.

---

## A: From Diagonal Covariance to Jacobian Orthogonality

The VAE fits a latent variable model $$p_\theta(x) =\int_z p_\theta(x\|z)p(z)$$ to  the data distribution $$p(x)$$ by maximising the Evidence Lower Bound (ELBO),

$$\ell(\theta, \phi) \quad =\quad \int p(x) \int q_\phi(z\|x) 
\ \{\ \log p_\theta(x\|z) \,-\, \beta \log \tfrac{q_\phi(z\|x)}{p(z)} \ \}\ dz dx\ ,$$

where the standard ELBO has $$\beta=1$$ ($$\beta>1$$ has been  found to improve disentanglement).[^betaVAE] 

> Note on ELBO: Maximising the ELBO can be viewed as ***maximum-likelihood$$^{++}$$***: maximising the likelihood $$\int p(x)\log p_\theta(x)$$ minimises the KL divergence between the data and model distributions, but this is often intractible for a latent variable model. Maximising the ELBO minimises the KL divergence between $$p(x)q_\phi(z\|x)$$ and $$p_\theta(x)p_\theta(z\|x)\doteq p_\theta(x\|z)p(z)$$, fitting two approximations of the joint distribution.

A Guassian VAE makes the following assumptons:
* $$p_\theta(x\|z) =\mathcal{N}(x;\,d(x),\sigma^2)\quad$$ with *decoder*  $$d$$ and fixed variance $$\sigma^2$$;
* $$q_\phi(z\|x)=\mathcal{N}(z;\,e(x),\Sigma_x)\quad$$ with *encoder* $$e$$ and learned variance $$\Sigma_x$$; and
* $$p(z)\quad\ \ \ =\mathcal{N}(z;\,0,I)\quad$$ where $$z_i$$ are *independent* with $$p(z_i)=\mathcal{N}(z_i;0,1)$$

Note that the VAE decoder $$d$$ maps latent variables $$z\in\mathcal{Z}$$ to  means $$\mu_z=\mathbb{E}[x\|z]\in \mathcal{X}$$, and if $$J_z$$ denotes $$d$$'s Jacobian (evaluated at $$z$$), $$J_{i,j} = \tfrac{\partial d(z)_i}{\partial z_j}$$ defines how a perturbation in the latent space (in direction $$z_j$$) translates to variation in the data space (in direction $$x_i$$).

Recent works show that diagonal posterior covariances ($$\Sigma_x$$) cause the Hessian of $$\log p_\theta(x\|z)$$ to be *approximately* diagonal, causing column orthgonality in $$J_z$$, which can be made more precise [(Opper & Achambeau, 2009)](http://www0.cs.ucl.ac.uk/staff/c.archambeau/publ/neco_mo09_web.pdf):

$$
\begin{equation}
  \Sigma_x 
    \ \ \overset{O\&A}{=}\ \ I - \mathbb{E}_{q(z\|x)}[\tfrac{\partial^2\log p_\theta(x\|z)}{\partial z_i\partial z_j}]
    \ \ \overset{\dagger}{\approx}\ \ I \,+ \tfrac{1}{\beta\sigma^2}J_z^\top J_z
  \tag{1}\label{eq:one}
\end{equation}
$$

Step $$\dagger$$ makes the assumption that the decoder's second derivative is small almost everywhere, e.g. as in a Relu network [(Abhishek & Kumar, 2020)](https://arxiv.org/pdf/2002.00041).

The ELBO is maximised if this relationship is achieved and so, if $$\Sigma_x$$ are diagnal, when **columns of $$J_z$$ are orthogonal**. Equivalently, the SVD $$J_z=U_zS_zV_z^\top$$ must have $$V_z=I$$, i.e. standard basis vectors $$e_i\in\mathcal{Z}$$ are right singular vectors of $$J_z$$ ($$\forall z$$).  Importantly, this means that variation in latent component $$z_i$$ corresponds to a variation in data space in direction $$u_i$$, the $$i^{th}$$ left singular vector of $$J_z$$ (i.e. column $$i$$ of $$U_z$$) with no affect in any other $$u_{j\ne i}$$.

**Take-away**: the ELBO is maximised if approximate posterior covariances match true posterior covariances, which can be expressed in terms of derivatives of $$p_\theta(x\|z)$$. This does not mean the Hessian is necessarily orthogonal, but if such solutions exists then the VAE tries to find them.
<!-- (hinting towards learning independent factors). -->

---

## B: From Orthogonality to Statistical Independence

Having seen that diagonal covariances promote column-orthogonality in the decoder Jacobian, the question is how this geometric proprety leads to the statistical property of disentanglement. To understand it, we consider the *push-forward* distribution defined by the decoder, which is supported over a manifold $$\mathcal{M}_d\subseteq\mathcal{X}$$. 

> A **push-forward distribution** describes the probability distribution of the output of a deterministic function given an input distribution. A VAE decoder latent samples of the prior $$p(z)$$ to the data space, defining a push-forward distribution  over $$\mathcal{M}_d$$. 

### Linear Case

<img src="/assets/disentanglement/linear.png" 
        alt="linear2" 
        width="440" 
        height="190" 
        class="image1" />


For intuition, we  consider the linear case $$x=d(z)=Dz$$,  $$D\in\mathbb{R}^{n\times m}$$, the model considered in [Probabilistic PCA (PPCA)](https://academic.oup.com/jrsssb/article-abstract/61/3/611/7083217), which has a tractible MLE solution and known optimal posterior

$$
\begin{equation}
  p_\theta(z\|x) = \mathcal{N}(z;\, \tfrac{1}{\sigma^2}M D^\top x,\, M) \quad\quad\quad M = (I + \tfrac{1}{\sigma^2}D^\top D)^{-1}
  \tag{2}\label{eq:two}
\end{equation}
$$

This can be seen as a special case of \eqref{eq:one}, thus the ELBO is maximised if $$\Sigma_x=M,\ \forall x\in\mathcal{X}$$, and using diagonal posteriors $$\Sigma_x$$ again implies $$V=I$$ (for SVD $$D=USV^\top$$). 

For a given point $$z^*\in \mathcal{Z}$$: 
* we define lines $$\mathcal{Z^{(i)}}\subset\mathcal{Z}$$ passing through $$z^*$$ parallel to each standard basis vector $$e_i$$, and their images under $$D$$, $$\mathcal{M}_D^{(i)}\subset\mathcal{M_d}$$ (lines following $$D$$'s left singular vectors); and
* consider $$u=U^\top x$$  ($$x$$ in the basis defined by columns of $$U$$), noting that: $$\tfrac{\partial u_i}{\partial z_j} =\{s_i \text{ if }i=j; 0 \text{ o/w}\}$$ (since $$\tfrac{dx}{dz} = \tfrac{dx}{du}\tfrac{du}{dz} = US$$ and $$\tfrac{dx}{du} = U$$).

The point here is to identify how independent dimensions $$z_i\in\mathcal{Z}$$ "flow" under the decoder. Indeed, by considering $$x$$ in the "$$U$$-basis", independent $$z_i$$ become independent components $$u_i$$, since it can be shown that:
1. $$\{u_i\}_i$$ are observations of *independent* random variables;
2. the push-forward of $$d$$ restricted to $$\mathcal{Z^{(i)}}$$ has density $$p(u_i) = s_i^{-1}p(z_i)$$ over $$\mathcal{M}_D^{(i)}$$;
3. the full push-forward satisfies $$p(Dz) = \|D\|^{-1}p(z) = \prod_i s_i^{-1}p(z_i) = \prod _ip(u_i)$$.

Altogether, this shows that the push-forward distribution generated by the decoder factorises as a product of independent univariate distributions ($$p(u_i)$$), each corresponding to a distinct latent dimension ($$z_i$$). Thus, if the data follows the assumed generateive process and itself factorises with unique factors (determined by ground truth $$s_i$$), then the ELBO is maximised when the independent components of the data and model distributions align and p(x) is **disentangled** as a product of *independent components* aligned with each latent dimension.

This is not so surprising in the linear case, since we know from the outset that the push-forward distribution is Gaussian and so necessarily factorises as a product of univariate Gaussians. However, we did not use that fact or rely on the linearity of $$d$$ at any step.

<!-- From $$J_z =\tfrac{d x}{d z}=US, \tfrac{d x}{d u} = U$$, we have $$\tfrac{d u}{d z} =S$$, so $$\tfrac{\partial u_i}{\partial z_j} = \{s_i$$ if $$i= j$$, otherwise $$0\}$$ and each $$u_i$$ depends on a distinct independent r.v. so are *independent*.
2. restricting $$D$$ to $$z_i\in \mathcal{Z}^{(i)}$$ and so $$u_i\in\mathcal{M_D^{(i)}}$$, $$p(u_i)= s_i^{-1}p(z_i)$$ (defined over $$\mathcal{M_D^{(i)}}$$)
 -->


### Non-linear Case with Diagonal Jacobian
<img src="/assets/disentanglement/non_linear.png" 
        alt="non_linear2" 
        width="420" 
        height="200" 
        class="image1" />



We now take an analogous approach for the general VAE ($$x=d(z)$$, $$d\in\mathcal{C}^2$$) with column-orthogonal decoder Jacobian. Notably, the Jacobian and its factors, $$J_z=U_zS_zV_z^\top$$, now depend on $$z$$, although column-orthgonality implies $$V_z=I,\ \forall z\in \mathcal{Z}$$ and $$U_z$$, $$S_z$$ are continuous w.r.t. $$z$$ since $$d\in\mathcal{C}^2$$. 

As previously, for a given point $$z^*\in \mathcal{Z}$$, we define lines $$\mathcal{Z^{(i)}}\subset\mathcal{Z}$$ passing through $$z^*$$ parallel to the standard basis (axis-aligned), and their images under $$d$$, $$\mathcal{M}_d^{(i)}\subset\mathcal{M_d}$$, which are potentially curved sub-manifolds, following (local) left singular vectors of $$D$$.

We again consider $$x$$ in the (local) basis defined by columns of $$U$$ as $$u=U^\top x$$ and still have $$\tfrac{\partial u_i}{\partial z_j} =\{s_i \text{ if }i=j; 0 \text{ o/w}\}$$.

As previously, we claim that independent dimensions $$z_i\in\mathcal{Z}$$ flow under the decoder to become independent components $$u_i$$ since:
1. $$\{u_i\}_i$$ are observations of *independent* random variables;
2. the push-forward of $$d$$ restricted to $$\mathcal{Z^{(i)}}$$ has density $$p(u_i) = s_i^{-1}p(z_i)$$ over $$\mathcal{M}_d^{(i)}$$;
3. the full push-forward satisfies $$p(d(z)) = \|J_z\|^{-1}p(z) = \prod_i s_i^{-1}p(z_i) = \prod _ip(u_i)$$.

Thus, following the same argument as in the linear case, the distribution over the decoder manifold factorises as a product of independent univariate push-foward distributions ($$p(u_i)$$), each corresponding to a distinct latent dimension ($$z_i$$). Again, if the true data distribution follows this generative process and so factorises and those factors are unique, then the ELBO is maximised when components of the model fit to those of the data and $$p(x)$$ is **disentangled** as a product of *independent components* aligned with each latent dimension. Each component is supported on a sub-manifold orthogonal to the others, capturing the variations along a single latent dimension.

We recommend reading [the full paper][the paper] for further details, such as:
* consideration of whether orthogonality is strictly _necessary_ for disentanglement (the argument above shows it is _sufficient_)
* _identifiability_ of the model, i.e. up to what symmetries can the VAE identify the ground truth generative factors
* the role of parameter $\beta$ in a [$\beta$-VAE][betVAE]
  * spoiler: $$\beta$$ is proportional to Var$$_\theta[x\|z]$$ where $$p_\theta(x\|z)$$ is of exponential family form (generalising $\sigma^2$ of a Gaussian-VAE).
  * $\beta$ determines how close data points need to be (in Euclidean norm) to be deemed similar and their representations merge.

<!-- 
**Interpreting β in β-VAEs**

β-VAEs introduce a hyperparameter β that scales the KL divergence term in the ELBO. Empirical studies have shown that increasing β enhances disentanglement, often at the cost of reconstruction quality. The sources offer a novel interpretation of β, viewing it as a factor scaling the variance of the likelihood distribution.

Increasing β corresponds to increasing the variance of the likelihood, essentially making the model more uncertain about its reconstructions. This increased uncertainty leads to greater overlap between the posterior distributions of nearby data points. As the ELBO encourages Jacobian orthogonality in expectation over the posterior distributions, larger overlap implies that orthogonality constraints apply over a broader region in the latent space, promoting stronger disentanglement.

Conversely, decreasing β reduces the likelihood variance, mitigating the issue of "posterior collapse". This phenomenon occurs when the likelihood becomes overly expressive, directly modeling the data distribution and rendering the latent variables redundant. By reducing the likelihood variance, β < 1 constrains the model's flexibility, preventing it from learning a trivial solution. -->

**Conclusion**

[Allen (2024)][paper] provides a compelling theoretical argument for the link between diagonal posterior covariance, Jacobian orthogonality, and disentanglement in VAEs. The analysis clarifies how a simple design choice, motivated by computational efficiency, leads to the learning of statistically independent data components. We hope that understanding this connection gives new insight into how VAEs work, how understanding of linear models may extend surprisingly well to non-linear models and may lead to new training algorithms that can more reliably achieve disentangelement.



---
### Notes

[paper]: https://arxiv.org/pdf/2410.22559
[betaVAE]: https://openreview.net/forum?id=Sy2fzU9gl
