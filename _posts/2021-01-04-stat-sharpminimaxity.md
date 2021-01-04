---
layout: post
title: "Bayesian shrinkage towards sharp minimaxity"
tags: [Statistics]
comments: true
---

Song, Q. (2020). Bayesian shrinkage towards sharp minimaxity. *Electronic Journal of Statistics*, ***14*(2)**, 2714-2741.

* 이 포스트는 위 논문의 주요 theoretical result를 요약/정리한 것이다.
* 시간이 허락한다면 아래 결과들에 대한 증명은 추후 포스트에 정리할 예정.



# 1. Introduction

### Settings

* 이 연구는 아래와 같은 sparse normal means model에 대한 것이다.  

$$
y^{(n)} = \theta^{(n)} + \epsilon, \\
\text{where }\enspace y^{(n)} =(y_1, \ldots, y_n)^T \in \mathbb R^n, \enspace \epsilon \sim \mathcal N(0, \sigma^2 I_n).
$$

* 논의의 간결함을 위해, W.L.O.G, let $\sigma^2 = 1$.
* True parameter $\theta^\ast$는 $s^{(n)}$개의 nonzero entry를 갖는 sparse vector.
* Asymptotically, $s^{(n)}$은 $n$이 증가함에 따라 증가한다고 가정.



### Shrinkage prior and optimality

* High dimensional model에 대한 Bayesian inference를 수행할 때 sparse한 추정을 위해 shrinkage prior를 사용한다.

* Sparsity를 유도하는 prior에는 크게 두 가지 종류

  * spike-and-slab prior (point-mass mixture prior)

  * (Global-local) shrinkage prior

* Classical spike-and-slab modeling은 모수의 차원이 커질 수록 mixture assignment에 대한 posterior computation을 수행하는 데 긴 시간이 걸리게 되고, mixing도 잘 되지 않음.

* 그에 반해 계산적 이점이 있는 shrinkage prior가 high dimensional setting에서의 sparse estimation에서 더 주목을 받음.

  * Bayesian Lasso, Horseshoe prior, Dirichlet-Laplace prior, Normal-Exponential-Gamma distribution, Generalized double Pareto distribution, Generalized Beta mixture of Gaussian distributions, etc.

* Shrinkage prior는 일반적으로 다음과 같이 나타낼 수 있다.
  * 전체적인 shrinkage의 정도를 나타내는 scale parameter $\tau$를 *global shrinkage parameter*라고 하자.
  * 이때 $\tau$는 특정 prior를 따르거나 혹은 deterministic한 값을 가질 수 있다.

$$
\pi(\theta \vert \tau ) = \prod_{i=1}^{n} \frac{1}{\tau}\pi_0\Big(\frac{\theta_i}{\tau}\Big), \quad \tau \sim \pi(\tau).
$$

* Shrinkage prior를 평가하는 기준이 되는 이론적 성질은 *posterior contraction rate*이다.

  * 예를 들어 $L_2$ posterior contraction rate은 다음을 만족하는 $r_n$으로 정의된다.
  
$$
\lim_{n \to \infty } \mathbb E_{\theta^\ast}\Big[ \pi(\Vert \theta- \theta^\ast\Vert \geq r_n \vert y )\Big] = 0, \text{ for any }\theta^\ast. \\
\iff \enspace \pi(\Vert \theta- \theta^\ast\Vert \geq r_n \vert y ) \to 0 \quad [\mathbb P_{\theta^\ast}]\enspace a.e., \text{ for any }\theta^\ast.
$$

  * 해당 prior를 주었을 때 (Bayes estimator 뿐만 아니라) 사후분포 자체가 얼마나 빨리 true parameter로 수렴하는가를 나타내는 개념.

* Normal means model에 대한 frequentist minimax rate은 다음과 같다.

$$
\inf_{\hat \theta} \sup_{\theta^\ast} \Vert \hat\theta - \theta^\ast \Vert = \sqrt{(2 + o(1)) s \log\Big(\frac{n}{s}\Big)}
$$

* 이 minimax rate와 같은 속도의 posterior contraction rate을 갖는 ***optimal*** shrinkage prior들이 제시되어왔다.

  * $\theta^\ast$에 대한 특정 조건 하에서, Dirichlet-Laplace prior는 다음 posterior contraction rate을 갖는다.

$$
r_n \asymp \sqrt{s \log \Big( \frac{n}{s}\Big)}
$$

  * Horseshoe prior는 임의의 $M_n \to \infty $에 대해 다음 posterior contraction rate을 갖는다.

$$
r_n = M_n \sqrt{s \log \Big( \frac{n}{s}\Big)}
$$

* 아래 최근 연구들에 의하면 $\pi_0$의 tail behavior가 posterior asymptotic에 중요한 역할을 하며, (near-) optimal posterior contraction rate을 얻기 위해서는 polynomially decaying인 $\pi_0$을 선택하는 것이 좋다.

  * van der Pas, S. L., Salomond, J. B., & Schmidt-Hieber, J. (2016). Conditions for posterior contraction in the sparse normal means problem. *Electronic journal of statistics*, ***10*(1)**, 976-1000.
  * Song, Q., & Liang, F. (2017). Nearly optimal Bayesian shrinkage for high dimensional regression. *arXiv preprint arXiv:1712.08964*.
  * Ghosh, P., & Chakrabarti, A. (2014). Posterior concentration properties of a general class of shrinkage priors around nearly black vectors. *arXiv preprint arXiv:1412.8161*.

* 실제로 Dirichlet-Laplace prior를 제외하면, 위에 예시로 소개된 shrinkage prior는 모두 polynomial tail을 갖는다.

* 지금까지 shrinkage prior에 대한 대부분의 연구는 $\pi_0$가 polynomial tail을 가질 때, (near-) optimal contraction을 위해 어떻게 global shrinkage parameter $\tau$를 선택할지에 대해 초점을 두고 있었다.

* 이 연구는 크게 다음과 같은 두 contribution을 갖는다.

  * $\pi_0$의 polynomial order가 1에 충분히 가깝다면, sharp한 Bayesian minimax를 달성할 수 있다.
    * 다만 이때 sharp minimaxity를 달성하기 위해서 $\tau$를 선택할 때 true sparsity ratio $(s/n)$의 정보가 필요한데, 이는 실제 분석 상황에서는 unknown.

$$
\frac{r_n}{\sqrt{2s \log \Big( \frac{n}{s}\Big)}}\enspace  \text{ sufficiently close to }1
$$

  * $\tau$에 대한 Beta modelling 방법을 제시한다.
    * 이는 unknown sparsity에 대해 adaptive하게 Bayesian sharply minimax inference를 가능케함.



# 2. Sharp Bayesian minimaxity

### Assumptions

* **[C.1]** The true model is sparse, $s = o(n)$.
* $s/n \to 0$, as $n \to \infty$.
* **[C.2]** The prior density $\pi_0(\cdot)$ is strictly decreasing on $(0,\infty)$ and increasing on $(-\infty, 0)$.
* **[C.3]** The tail of $\pi_0(\cdot)$ is polynomially decaying with polynomial order $\alpha > 1$, i.e., there exist some positive constants $M$ and $C_2 > C_1$ such that for any $\vert \theta \vert > M$, 

$$
C_1 \vert \theta\vert^{-\alpha}\leq \pi_0(\theta) \leq C_2 \vert \theta\vert^{-\alpha}
$$

* 우선 global shrinkage parameter $\tau$가 deterministic하다고 가정하자.
  * 이는 뒤에서 Beta modelling을 이용해 해제할 가정.



## Theorem 2.1.

### Statement

*Let a positive constant $\omega$ be given, and $\tau^{\alpha - 1} \geq (s/n)^c \sqrt{\log(n/s)}$ for some $c \in (0,1+\omega/2)$,*

* *if $\tau^{\alpha - 1} \prec \{(s/n)\log(n/s)\}^\alpha$, then*

$$
\lim_{n \to \infty} \mathbb E_{\theta^\ast}\Big[ \pi\Big(\Vert \theta- \theta^\ast\Vert \geq C_1(\omega) \sqrt{s\log(n/s)} \Big\vert D_n \Big)\Big] = 0, \tag{2.1}
$$

*where $C_1(\omega) = \sqrt{2+\omega} + \sqrt{\omega}$.*

* *If furthermore, $\tau^{\alpha - 1} \prec (s/n)^\alpha \{\log(n/s)\}^{(\alpha+1)/2}$, then*

$$
\lim_{n \to \infty} \mathbb E_{\theta^\ast}\Big[ \pi\Big(\Vert \theta- \theta^\ast\Vert_1 \geq s \cdot C_2(\omega) \sqrt{\log(n/s)} \Big\vert D_n \Big)\Big] = 0, \tag{2.2}
$$

*where $C_2(\omega) = \sqrt{2+\omega} +\sqrt{\omega^2/5}+ \sqrt{\omega/5}$.* 



### Remarks

* $C_1(\omega), C_2(\omega) \searrow \sqrt{2}$ as $\omega \searrow 0$.

* $\tau$에 대한 첫 번째 조건은 $\tau$가 너무 작지 않게 하는 조건.

  * 과도하게 작은 $\tau$는 nonzero $\theta_i$에 대해 과한 shrinkage를 부과하게 되는 문제.

* $\tau$에 대한 두 번째 조건은 $\tau$가 너무 크지 않게 하는 조건.

* 큰 $\tau$는 true zero $\theta_i$에 대해 shrinkage를 충분히 주지 못하게 되는 문제.

* 이 두 조건을 만족시키기 위해 $\alpha \in (1, 1+\omega/2)$를 골라야 한다.

  * 아래 부등식의 좌변은 $n$이 커지면 $\infty$로 발산.

  $$
  \frac{\{(s/n)\log(n/s)\}^\alpha}{\tau^{\alpha - 1}} \leq (s/n)^{\alpha-c} \{\log(n/s)\}^{\alpha-\frac{1}{2}}
  $$

  * 그러기 위해서는 $\alpha - c \leq 0, \text{ }\alpha \leq c$ 이어야 함.
  * 조건에 의해 $c \in (0, 1+\omega/2 )$이므로, $\alpha \leq c < 1+\omega/2$.
  * 따라서, $1 < \alpha < 1+\omega/2$.

* 임의의 polynomially decaying shrinkage prior는 위 정리의 조건에 맞게 global shrinkage $\tau$를 잘 결정하면 optimal Bayesian contraction rate $O(\{s\log(n/s)\}^{1/2})$을 달성하게 된다.

  * 이는 Horseshoe prior에 대해 기존에 증명된 사실보다 더 optimal rate에 가까운 결과.

$$
O(M_n\{s\log(n/s)\}^{1/2}) \text{ with }M_n \to \infty.
$$

* 또한 위 정리를 통해 Bayesian contraction rate의 multiplicative constant가 $\pi_0(\cdot)$의 polynomial order와 positively related임을 알 수 있다.

  * $1 < \alpha < 1+\omega/2$ 이므로, $\alpha$가 커지면 위 정리의 조건을 만족하는 $\omega$의 범위의 하한도 올라간다.
  * $\omega$가 커지면 $C_1(\omega), C_2(\omega)$는 커지므로, $\alpha$와 contraction rate의 multiplicative constant $C_1(\omega), C_2(\omega)$는 positively related.



## Corollary 2.1.

### Statement

*If $\alpha \leq 1+ \omega/2$ and $\tau^{\alpha - 1} \asymp (s/n)^c$ for some $c \in [\alpha , 1+ \omega/2)$, then (2.1) and (2.2) hold.*



### Remarks

* 임의의 작은 상수 $c>0$에 대해, $\log(n/s)$는 점근적으로 $(n/s)^c$에 dominate되기 때문에 위 Theorem 2.1에서 제시된 $\tau$에 대한 조건은 간소화될 수 있다.
  * $\log(n/s)$를 $(n/s)$에 아주 작은 지수를 둔 것으로 대체함으로써.
  * **첫 조건 $\tau^{\alpha - 1} \geq (s/n)^c \sqrt{\log(n/s)}$는 따라서 무시가능한가?**
* 아래 두 조건은 $\tau^{\alpha - 1} \asymp (s/n)^{c}$에 대해 $\alpha \leq c$로 선택하면 만족. 따라서 $c \in [\alpha, 1+\omega/2)$.
  * $\tau^{\alpha - 1} \prec \{(s/n)\log(n/s)\}^\alpha$
  * $\tau^{\alpha - 1} \prec (s/n)^\alpha \{\log(n/s)\}^{(\alpha+1)/2}$

* 이 corollary는 Theorem 2.1보다 $\tau$에 대해 보다 간소화된 조건을 제시.



## Theorem 2.2 

### Statement

*Let a positive constant $\omega$ be given, and $\tau^{\alpha - 1} \geq (1/n)^c \sqrt{\log(n/s)}$ for some $c \in (0,1+\omega/2)$. If $\tau^{\alpha - 1} \prec (s/n)^\alpha \{\log(n/s)\}^{(\alpha+1)/2}$, then*

$$
\begin{align*}
&\lim_{n \to \infty} \mathbb E_{\theta^\ast}\Big[ \pi\Big(\Vert \theta- \theta^\ast\Vert \geq C_1(\omega) \sqrt{s\log(n)} \Big\vert D_n \Big)\Big] = 0,  \\
&\lim_{n \to \infty} \mathbb E_{\theta^\ast}\Big[ \pi\Big(\Vert \theta- \theta^\ast\Vert_1 \geq s \cdot C_2(\omega) \sqrt{\log(n)} \Big\vert D_n \Big)\Big] = 0, \end{align*}\tag{2.3}
$$

*for the same functions $C_1(\omega), C_2(\omega)$ used in Theorem 2.1.*



### Remarks

* Theorem 2.1과 이를 간소화한 Corollary 2.1에서는 $\tau$를 아래와 같이 선택하도록 했다.
  * 이때 $\delta$는 nonnegative small value.

$$
\tau \asymp \Big(\frac{s}{n}\Big)^{(\alpha + \delta)/(\alpha - 1)}
$$

* In practice, 우리는 true sparsity $s$를 사전에 알지 못한다는 문제가 있다.
  * 뒤의 Section 3에서 unknown sparsity에 대해 adaptive한 full Bayesian approach를 소개한다.
* Theorem 2.2는 $\tau$를 아래와 같이 선택했을 때의 contraction rate에 대한 결과를 소개한다.
  * 마찬가지로 이때 $\delta$는 nonnegative small value.

$$
\tau \asymp \Big(\frac{1}{n}\Big)^{(\alpha + \delta)/(\alpha - 1)}
$$

* 이때의 Bayesian contraction rate는 다음과 같다.
  * $L_2$ rate : $\sqrt{s\log(n)}$
  * $L_1$ rate : $s\sqrt{\log(n)}$

* True sparsity $s$의 rate의 여러 경우에 따라 이는 optimal 혹은 suboptimal rate이 된다.
  * If $\log s \prec \log n$, above rates are asymptotically same with the rate of Theorem 2.1.
  * If $s \asymp n^c$ for some $c \in (0,1)$, above rates have same order with the rates of Theorem 2.1.
    * But with large multiplicative constant.
  * If $\log s \sim \log n$, e.g., $s = n / \log n$, above rates are strictly greater than the rates of Theorem 2.1.
* 다만, 엄밀하게 보았을 때 Theorem 2.1, 2.2에 의해 얻은 결과는 posterior contraction rate의 **upper bound**이므로, Theorem 2.2에서와 같이 $\tau$를 선택한 prior specification이 반드시 suboptimal posterior convergence를 야기한다고 결론내릴 수는 없다.
  * $r_n$이 Bayesian contraction rate이면, $s_n \geq r_n$인 $s_n$ 역시 Bayesian contraction rate이므로.
  * 즉 이 rate보다 느리게 감소하는 rate은 posterior consistency가 만족한다는 (rate가 0으로 감소하는 속도의) **upper bound**를 제시한 것.





# 3. Adaptive Bayesian inference

* Global shrinkage parameter $\tau$를 결정하기 위해 Section 2에서 제시한 방법은 true sparsity $s$가 prespecified hyperparameter라는 한계가 있다.
* 따라서 실제 분석 상황에서는 $\tau$를 결정할 adaptive한 방법이 필요.
* Bayesian paradigm에서 hyperparameter를 다루는 방법에는 크게 두 가지 방법이 있다.
  * Empirical Bayes
  * Full Bayesian approach : 이 논문에서는 이 방법을 사용

* Theorem 2.1은 $\tau^{\alpha - 1} \prec \{(s/n)\log(n/s)\}^\alpha$, 즉 $n$이 증가함에 따라 $\tau$가 0으로 감소하도록 조건을 설정했다.
  * $n$이 증가함에 따라 prior $\pi(\tau)$가 stochastically decreasing하도록 설계해야.
* 그러면서도 너무 빠르지 않은 적당한 rate으로 $\pi(\tau)$가 0으로 shrink해야.
  * 이는 $\tau^{\alpha - 1} \geq (1/n)^c \sqrt{\log(n/s)}$의 조건을 반영한 것.



## Theorem 3.1.

### Statement

*If $\alpha \leq 1 + \omega /2$ , $\pi(\tau) $ satisfies that:* 

$$
\begin{align*}
&-\log\pi\Big\{\Big(\frac{s}{n}\Big)^{(1+\omega/2) / (\alpha - 1)} \leq \tau \leq\Big(\frac{s}{n}\Big)^{\alpha/(\alpha - 1) } \Big\} \prec s \log \Big(\frac{n}{s}\Big),\\
&-\log\pi\Big\{ \tau \geq \Big(\frac{s}{n}\Big)^{\alpha/(\alpha - 1) } \Big\} \succ s \log \Big(\frac{n}{s}\Big),
\end{align*}
$$

*and $\max_j \vert \theta_j^\ast\vert \leq ({n}/{s})^{\omega/(5\alpha)}$, then (2.1) and (2.2) still hold.*



### Remarks

* 이 정리는 1에 충분히 가까운 $\alpha$를 고르고 $\tau$에 대한 prior $\pi(\tau)$를 적절하게 특정함으로써, Theorem 2.1의 sharp minimaxity를 여전히 얻을 수 있다는 사실을 보여준다.
* $\pi(\tau)$에 대한 조건은 각각 다음을 의미한다.
  * $\tau$의 optimal range $[(s/n)^{(1+\omega/2) / (\alpha - 1)} ,(s/n)^{\alpha/(\alpha - 1) }]$에 부여되는 확률의 lower bound.
  * $\pi(\tau)$가 0 주변으로 점점 concetrated.

* 위 두 조건을 만족하는 $\pi(\tau)$로는 다음과 같은 사전분포를 고려해볼 수 있다.
  * $\delta = 1-\alpha/c(\alpha - 1)$
  * $\delta^\prime = -1+((1+\omega/2)/c(\alpha - 1))$

$$
\tau = \tau_0^c, \quad \tau_0 \sim \Beta(1,n) \enspace \text{for some }c \in\Big(\frac{\alpha}{\alpha-1}, \frac{1+\omega/2}{\alpha - 1}\Big).
$$

$$
\begin{align*}
\pi\Big\{ \tau \geq \Big(\frac{s}{n}\Big)^{\alpha/(\alpha - 1) } \Big\} &= \pi\Big\{ \tau_0 \geq \Big(\frac{s}{n}\Big)^{\alpha/c(\alpha - 1) } \Big\} \\
&=\Big\{1-\Big(\frac{s}{n}\Big)^{\alpha/c(\alpha - 1)} \Big\}^n \\
&\sim \exp\Big( -s^{\alpha/c(\alpha - 1)}n^{1-\alpha/c(\alpha - 1)} \Big) \\
&\sim \exp\Big( -s\Big(\frac{n}{s}\Big)^{1-\alpha/c(\alpha - 1)} \Big) \\
&\sim \exp\Big( -s\Big(\frac{n}{s}\Big)^{\delta} \Big)
\end{align*}
$$

$$
\begin{align*}
&\pi\Big\{\Big(\frac{s}{n}\Big)^{(1+\omega/2) / (\alpha - 1)} \leq \tau \leq\Big(\frac{s}{n}\Big)^{\alpha/(\alpha - 1) } \Big\} \\
&=\pi\Big\{\Big(\frac{s}{n}\Big)^{(1+\omega/2) / c(\alpha - 1)} \leq \tau_0 \leq\Big(\frac{s}{n}\Big)^{\alpha/c(\alpha - 1) } \Big\}  \\
&=\Big\{1-\Big(\frac{s}{n}\Big)^{(1+\omega/2) / c(\alpha - 1)} \Big\}^n - \Big\{1-\Big(\frac{s}{n}\Big)^{\alpha/c(\alpha - 1) } \Big\}^n \\
&\sim \exp\Big( -n\Big(\frac{s}{n}\Big)^{(1+\omega/2)/c(\alpha - 1)} \Big) -  \exp\Big( -n\Big(\frac{s}{n}\Big)^{\alpha/c(\alpha - 1)} \Big) \\
&\sim \exp\Big( -s\Big(\frac{n}{s}\Big)^{1-((1+\omega/2)/c(\alpha - 1))} \Big) \\
&\sim \exp\Big( -s\Big(\frac{n}{s}\Big)^{-\delta^\prime} \Big)
\end{align*}
$$

* True parameter $\theta^\ast$에 대해서 부과된 조건 $\max_j \vert \theta_j^\ast\vert \leq ({n}/{s})^{\omega/(5\alpha)}$는 다음과 같이 고쳐쓸 수 있다.
  * 이때 조건 **[C.1]**에 의해 $\log(n/s) \to \infty $이므로, $n$이 증가함에 따라 true signal $\theta^\ast$의 strength가 커지는 것을 허용하는 조건임을 알 수 있다.
  * 만약 $\log (\max_j \vert \theta_j^\ast\vert) = o(\log(n/s))$, 즉 maximum true signal이 sub-polynomial하게 커진다면, $\omega$를 충분히 0에 가깝게 작게 잡아 sharp minimaxity를 얻을 수 있다.
  * 만약 어떤 $a>0$에 대해 $\max_j \vert \theta_j^\ast\vert \asymp (n/s)^a$, 즉 maximum true signal이 polynomial하게 커진다면, $\omega$를 0에 가깝게 줄이지 못하여 Bayesian contraction rate의 multiplicative constant $C_1(\omega), C_2(\omega)$를 arbitrary하게 줄이는 sharp minimaxity는 달성할 수 없다. 
    * 하지만 이때도 여전히 contraction rate는 minimax rate인 $O(\{s\log(n/s)\}^{1/2})$을 달성한다.
    
$$
\frac{\log (\max_j \vert \theta_j^\ast\vert)}{\log(n/s)} \leq \frac{\omega}{5\alpha}.
$$

* Dirichlet-Laplace prior는 optimality를 보이는 데 있어, true parameter value 주변에서 prior가 특정 확률 이상을 부여한다는 조건 외에도, $\Vert \theta^\ast\Vert $에 upper bound를 주는 조건을 필요로 한다.

$$
\Vert \theta^\ast \Vert \leq \sqrt{s} (\log n)^2
$$

  * $L_1$-norm과 $\max_j \vert \theta_j^\ast\vert$를 기준으로 이는 다음을 나타낸다. (C-S inequality: $\Vert \theta \Vert_1^2 \leq s \Vert \theta \Vert^2$)

$$
\max_j \vert \theta_j^\ast\vert \leq\Vert \theta^\ast \Vert_1 \leq s (\log n)^2
$$

  * 이는 본 연구의 조건보다 더 강한 조건



### Theorem 3.2.

### Statement 

*If $\alpha \leq 1 + \omega /2$ , $\pi(\tau) $ satisfies:* 

$$
\begin{align*}
&-\log\pi\Big\{\Big(\frac{s}{n}\Big)^{(1+\omega/2) / (\alpha - 1)} \leq \tau \leq\Big(\frac{s}{n}\Big)^{\alpha/(\alpha - 1) } \Big\} \prec s \log \Big(\frac{n}{s}\Big),\\
&-\log\pi\Big\{ \tau \geq \Big(\frac{s}{n}\Big)^{\alpha/(\alpha - 1) } \Big\} \succ s \log \Big(\frac{n}{s}\Big),
\end{align*}
$$

*and the prior of $\tau$ has support as follows:*

$$
\Big[\Big(\frac{1}{n}\Big)^{c/(\alpha-1)}, \infty\Big)
$$

*then (2.3) holds.*



### Remarks

* Optimal rate을 증명하기 위해 Theorem 3.1은 true parameter $\theta^\ast$에 대해 아래와 같은 조건을 추가로 주었다.

$$
\max_j \vert \theta_j^\ast\vert \leq ({n}/{s})^{\omega/(5\alpha)}
$$

* 이는 $\tau$에 prior를 부여한 full Bayesian 방법을 사용함으로써 각 $\theta_i$들이 더이상 independent가 아니게 된 사실에 기인한 것으로 해석할 수 있다. **(왜?)**
* 아래 연구에서는 Horseshoe prior의 global shrinkage parameter $\tau$에 대해 truncated half Cauchy prior를 주고, $\theta_j^\ast$에 대한 가정을 하지 않고 suboptimal contraction rate을 증명했다.
  * van der Pas, S., Szabó, B., & van der Vaart, A. (2017). Adaptive posterior contraction rates for the horseshoe. *Electronic Journal of Statistics*, ***11*(2)**, 3196-3225.
* 이와 유사하게, Rate의 optimality를 포기하는 대신 true parameter에 대한 가정을 하지 않는 이론적 결과를 이끌어낸 것이 Theorem 3.2이다.
