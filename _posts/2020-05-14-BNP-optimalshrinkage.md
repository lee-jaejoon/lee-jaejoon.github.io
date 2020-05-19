---
layout: post
title: "Dirichlet–Laplace priors for optimal shrinkage"
tags: [Bayesian Nonparametrics]
comments: true
---

# 1. Setting

$$
y_i = \theta_i + \varepsilon_i, \quad \varepsilon_i \sim \text{N}(0,1), \quad 1 \leq i \leq n. \tag{1}
$$

$$
\begin{align*}
\theta \sim \text{DL}_{a_n} \iff \theta_j \vert \phi_j , \tau &\sim \text{DE}(\phi_j \tau) \\
\phi &\sim \text{Dir}(a, \ldots , a) \tag{2}\\
\tau &\sim \Gamma(na, 1/2)
\end{align*}
$$

Or equivalently,

$$
\begin{align*}
\theta \sim \text{DL}_{a_n} \iff \theta_j \vert \psi_j &\sim \text{DE}(\psi_j) \\
\psi_j &\sim \Gamma(a, 1/2) \tag{3}
\end{align*}
$$


# 2. Statement of the Theorem

이 정리는 *Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015). Dirichlet–Laplace priors for optimal shrinkage. Journal of the American Statistical Association.*의 **Theorem 3.1**이며, 위와 같이 정의한 Dirichlet-Laplace shrinkage prior $\text{DL}_{a_n}$가 **optimal posterior contraction rate**을 갖는다는 것을 보인 theorem이다.


*Consider model (1) with $\theta \sim \text{DL}_{a_n}$ as in (2), where $a_n = n^{-(1+\beta)}$ for some $\beta >0$ small. Assume $\theta_0 \in l_0[q_n;n]$ with $q_n = o(n)$ and $\Vert \theta_0 \Vert_2^2 \leq q_n \log^4 n$. Then, with $s_n^2 = q_n \log(n/q_n)$ and for some constant $M>0$,*

$$
\lim_{n \rightarrow \infty} E_{\theta_0} \Pi_n( \Vert \theta - \theta_0 \Vert_2 < M s_n \vert y^{(n)} ) = 1 \tag{4}
$$

*If $a_n = 1/n$ instead, then (4) holds when $q_n \gtrsim \log n$.*



# 3. Proof

논문에서 제시된 증명과 같이, 증명과정의 중간 중간 적절하게 설정해주는 상수들은 서로 구분 없이 $C$로 표기한다. 일부 논문의 계산 오류로 추정되는 것을 수정한 부분이 있지만, 흐름 상으로는 논문의 증명을 벗어나지 않는다.



## 1) $P_{\theta_0}(\mathcal A_n^c) \leq e^{-r_n^2/2}$

이 부분은 *Castillo and van der Vaart (2012)*의 **Lemma 5.2**에 대한 증명과정이다.

먼저 다음과 같이 사건 $\mathcal A_n  \in \sigma(y^{(n)})$을 정의한다. 이 때 $\sigma(y^{(n)})$는 $y^{(n)}$에 의해 생성된 sigma-algebra이다.

$$
\mathcal D_n = \int\prod_{i=1}^n \frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta),\\
\mathcal A_n = \{ \mathcal D_n \geq e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)\}
$$

$\tilde \Pi \leq \Pi$를 만족하는 positive measure $\tilde \Pi$에 대해 다음이 만족한다. 여기서는 $\tilde \Pi$를 ball $\{ \Vert \theta - \theta_0 \Vert < r\}$ 위로 $\Pi$를 제한시킨 measure로 정의한다.

$$
\begin{align*}
\log \frac{\mathcal D_n}{\Vert \tilde \Pi\Vert} &= \log \int\prod_{i=1}^n \frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} \frac{1}{\Vert \tilde \Pi\Vert}d\Pi(\theta) \\
&\geq \log \int\prod_{i=1}^n \frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} \frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) \\
&\geq \int \log \prod_{i=1}^n \frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} \frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) \\
&= \int \log \frac{\exp(-\frac{1}{2}(y - \theta)^T(y-\theta))}{\exp(-\frac{1}{2}(y - \theta_0)^T(y-\theta_0))} \frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) \\
&= \int -\frac{1}{2}(-2y^T\theta + \theta^T \theta + 2y^T \theta_0 - \theta_0^T \theta_0) \frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) \\
&= \int (y - \theta_0)^T(\theta - \theta_0)\frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) - \frac{1}{2}\int (\theta - \theta_0)^T(\theta - \theta_0)\frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) \\
&= (y - \theta_0)^T \tilde \mu - \frac{1}{2} \tilde \sigma^2 \\
\end{align*}
$$

이 때 세 번째 줄의 부등호는 Jensen 부등식을 사용한 것이고, $\tilde \mu, \tilde \sigma^2$는 다음과 같이 정의한다.

$$
\tilde \mu = \int (\theta - \theta_0)\frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) \\
\tilde \sigma^2 = \int (\theta - \theta_0)^T(\theta - \theta_0)\frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta)
$$

이를 간단히 나타내면,

$$
\mathcal D_n \geq \Vert \tilde \Pi \Vert \exp\Big(\tilde \mu^T (y-\theta_0) - \frac{1}{2}\tilde \sigma^2 \Big) \tag{5}
$$

$\tilde \mu, \tilde \sigma^2$는 다음과 같은 성질을 만족한다.

$$
\Vert \tilde \mu \Vert  = \frac{\Vert \int (\theta - \theta_0)d\tilde \Pi(\theta) \Vert}{\Vert \tilde \Pi\Vert} \leq \frac{ \int \Vert\theta - \theta_0 \Vert d\tilde \Pi(\theta) }{\Vert \tilde \Pi\Vert} \leq \frac{ r\int d\tilde \Pi(\theta) }{\Vert \tilde \Pi\Vert} = r \\
\tilde \sigma^2 = \int \Vert \theta - \theta_0\Vert^2 \frac{1}{\Vert \tilde \Pi\Vert}d\tilde \Pi(\theta) \leq r^2 \frac{ \int d\tilde \Pi(\theta) }{\Vert \tilde \Pi\Vert}=r^2
$$

따라서, $r = 2r_n$으로 설정하면,

$$
\begin{align*}
P_{\theta_0}(\mathcal A_n^c) &= P_{\theta_0}\Big( \mathcal D_n < e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n )\Big) \\
&= P_{\theta_0}\Big( \mathcal D_n < e^{-4r_n^2} \Vert \tilde \Pi \Vert\Big) \\
&\leq P_{\theta_0}\Big( \Vert \tilde \Pi \Vert \exp(\tilde \mu^T (y-\theta_0) - \tilde \sigma^2/2 )  < e^{-4r_n^2} \Vert \tilde \Pi \Vert\Big) \\
&= P_{\theta_0}\Big( e^{\tilde \mu^T (y-\theta_0) - \tilde \sigma^2/2 }  < e^{-4r_n^2} \Big) \\
&= P_{\theta_0}\Big( \tilde \mu^T (y-\theta_0) - \tilde \sigma^2/2  < -4r_n^2 \Big) \\
&= P_{\theta_0}\Big( \tilde \mu^T (y-\theta_0)   < -\frac{r^2}{2}\Big)
\end{align*}
$$

세 번째 줄의 부등호는 식 (5)의 결과에 의한 것이다. 

$y$가 $P_{\theta_0}$, 즉 $\text{N}(\theta_0, I)$를 따른다고 할 때, 표준정규분포를 따르는 확률변수 $Z$에 대해 다음이 성립한다.

$$
\tilde \mu^T (y-\theta_0) \stackrel{d}{=} Z \Vert \tilde \mu \Vert \sim \text{N}(0, \Vert \tilde \mu\Vert^2)
$$

$$
\begin{align*}
P_{\theta_0}(\mathcal A_n^c) &\leq P_{\theta_0}\Big( \tilde \mu^T (y-\theta_0)   < -\frac{r^2}{2} \Big) \\
&= P_{\theta_0}\Big( Z   < -\frac{r^2}{2 \Vert \tilde \mu \Vert}\Big) \\
&\leq P_{\theta_0}\Big( Z   < -\frac{r}{2}\Big) \\
&\leq e^{-{r^2}/8} \\
&= e^{-{r_n^2}/2} \\
\end{align*}
$$



## 2) Construction of covering sets of parameter space

$\mathcal S_n$은 다음과 같이 정의된 $\{ 1, 2,\ldots, n\}$의 부분집합의 모임이다. 이 때 $S \in \mathcal S_n$는 $\theta $가 nonzero 값을 갖는 index를 의미한다.

$$
\mathcal S_n = \Big\{ S \subset \{1, 2, \ldots , n\} : \vert S \vert \leq Aq_n \Big\}
$$

한 $S \in \mathcal S_n$와 양의 정수 $j$에 대해, 다음과 같이 $\Theta_{S,j,n}$을 정의하자.

$$
\Theta_{S,j,n} = \Big\{ \theta \in \mathbb R^n : \text{supp}_{\delta_n}(\theta) = S, 2jr_n \leq \Vert \theta-\theta_0 \Vert_2 \leq 2(j+1)r_n\Big\}
$$

이 때 $\delta_n = r_n /n$으로 둔다. 그리고 $\Theta_{S,j,n}$의 $jr_n$ net을 $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$와 같이 나타내자. 이 때 다음이 만족한다.

$$
\forall \theta \in \Theta_{S,j,n}, \quad \exists i \enspace \text{ s.t. } \Vert \theta - \theta^{S,j,i}\Vert < jr_n
$$

그럼 이와 같은 $\Theta_{S,j,n}$의 $jr_n$ net $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$는 어떻게 만들 수 있을까?

$\vert S\vert$-dimensional ball $\{\Vert \phi - \theta_{0S}\Vert  \leq 2(j+1)r_n\}$의 $\frac{1}{2}jr_n$ net을 $\{ \phi^{S,j,i} : i = 1, \ldots, N_{S,j}\}$라고 하자. Covering number의 volume argument에 의해, 어떤 상수 $C$에 대해 $N_{S,j}$가 다음을 만족하도록 위의 $\vert S\vert$-dimensional ball에 대한 $jr_n$ net을 잡을 수 있다. ([Link](https://www.stat.berkeley.edu/~bartlett/courses/2013spring-stat210b/notes/12notes.pdf) 참고)

$$
N_{S,j} \leq C^{\vert S\vert}
$$

$\theta_S^{S,j,i} = \phi^{S,j,i}$, $\theta_k^{S,j,i} = 0$ for $k \in S^c$와 같이 $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$을 정의하고, 이것이 $\Theta_{S,j,n}$의 $jr_n$ net이 됨을 보이자. 임의의 $\theta \in \Theta_{S,j,n}$를 고르고, 그에 대해 $\Vert \theta_S^{S,j,i} - \theta_S \Vert \leq \frac{1}{2}jr_n$을 만족하는 $1 \leq i\leq N_{S,j}$를 고르자.

$$
\begin{align*}
\Vert \theta^{S,j,i} -\theta \Vert_2^2 &= \left\Vert \theta^{S,j,i}_S -\theta_S \right\Vert_2^2 + \left\Vert \theta_{S^c} - \mathbf 0 \right\Vert \\
&= \left\Vert \phi^{S,j,i} -\theta_S \right\Vert_2^2 + \left\Vert \theta_{S^c}\right\Vert \\
&\leq \Big(\frac{1}{2}jr_n\Big)^2 + (n-\vert S \vert)\delta_n^2 \\
&\leq \frac{1}{4}j^2r_n^2 + (n-\vert S \vert)\frac{r_n^2}{n^2} \\
&\leq r_n^2\Big(\frac{j^2}{4}+ \frac{1}{n} - \frac{\vert S \vert}{n^2}\Big) \\
&\leq j^2 r_n^2
\end{align*}
$$

세 번째 줄의 부등호는 $\Theta_{S,j,n}$의 $S$에의 projection이 중심 $\theta_{0S}$, 반지름 $2(j+1)r_n$의 $\vert S\vert$-dimensional ball의 부분집합이라는 사실에 따른 것이다. 또한 다섯 번째 줄의 부등호는 $n$에 대한 이차방정식으로 놓고 판별식을 계산하면 $\vert S\vert > 1/3$이 만족할 때 모든 $n$에 대해 만족한다는 사실을 알 수 있고, $A$를 적절하게 크게 잡으면 이는 문제가 없다.

따라서 위와 같이 construct한 $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$은 $\Theta_{S,j,n}$의 $jr_n$ net이 된다. 따라서 $\theta^{S,j,i}$를 중심으로 갖는 $S$ 위의 반지름 $jr_n$의 ball, $B_{S,j,i}$의 union은 $\Theta_{S,j,n}$을 덮는다(cover). 

$$
B_{S,j,i} = \Big\{ \theta : \text{supp}_{\delta_n}(\theta) = S ,\Vert \theta - \theta^{S,j,i} \Vert < jr_n\Big\}\\
\Theta_{S,j,n} \subset \bigcup_{i = 1}^{N_{S,j}} B_{S,j,i}
$$



## 3) $E_{\theta_0} \Pi_n(\vert \text{supp}_{\delta_n}(\theta )\vert > Aq_n \vert y^{(n)}) \rightarrow 0$

이 부분은 논문의 **Theorem 3.2**에 소개된 내용이다. Statement는 다음과 같다. 이는 $\text{DL}_{a_n}$ prior 하에서 구한 사후분포에서, $\theta$의 sparsity level(nonzero entry의 수)이 true $\theta_0$의 sparsity level, $q_n$의 특정 상수배($\times A$)를 넘을 사후확률이 $n$이 증가함에 따라 $0$에 수렴한다는 **posterior compressibility**를 나타내는 정리이다.

*Consider model (1) with $\theta \sim \text{DL}_{a_n}$ as in (2), where $a_n = n^{-(1+\beta)}$ for some $\beta >0 $ small. Assume $\theta_0 \in l_0(q_n ; n)$ with $q_n = o(n)$. Let $\delta_n = q_n / n$. Then,*

$$
\lim_{n \rightarrow \infty} E_{\theta_0} \Pi_n(\vert \text{supp}_{\delta_n}(\theta) \vert > A q_n \vert y^{(n)}) = 0,
$$

*for some constant $A>0$. If $a_n = 1/n$ instead, then it holds when $q_n \gtrsim \log n$.*



증명은 다음과 같다. 먼저 true $\theta_0$의 support $\text{supp}(\theta_0)$를 $\text{supp}(\theta_0)= S_0$라고 하자.

$$
\begin{align*}
&E_{\theta_0} \Pi_n(\vert \text{supp}_{\delta_n}(\theta) \vert > A q_n \vert y^{(n)}) \\
&= E_{\theta_0} \Pi_n(\vert \text{supp}_{\delta_n}(\theta) \cap S_0\vert + \vert \text{supp}_{\delta_n}(\theta) \cap S_0^c\vert > A q_n \vert y^{(n)}) \\
&\leq E_{\theta_0} \Pi_n( \vert \text{supp}_{\delta_n}(\theta) \cap S_0^c\vert > (A-1) q_n \vert y^{(n)}) \\
\end{align*}
$$

마지막 줄의 부등호는 $\vert \text{supp}_{\delta_n}(\theta) \cap S_0\vert \leq q_n \text{ a.s.}$ 때문에 성립하는 것이다. 따라서 우리는 다음을 보이고자 한다.

$$
\text{WTS: }E_{\theta_0} \Pi( \vert \text{supp}_{\delta_n}(\theta) \cap S_0^c\vert > A^\prime q_n \vert y^{(n)}) \rightarrow 0, \quad \text{ as } n \rightarrow \infty.
$$

이하의 과정에서는 표현의 편의상 $A^\prime = A-1$를 $A$로 표기한다. 다음과 같이 사건 $\mathcal B_n$을 정의한다.

$$
\mathcal B_n =  \Big\{ \theta : \vert \text{supp}_{\delta_n}(\theta) \cap S_0^c\vert > A q_n \Big\}
$$

사건 $\mathcal B_n$의 사후확률은 다음과 같다.

$$
\begin{align*}
\Pi_n( \mathcal B_n \vert y^{(n)}) &=\frac{\int_{ \mathcal B_n} \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}{\int \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)} \\
&=\frac{\int_{ \mathcal B_n} \prod_{i \in S_0^c}f_{\theta_i}(y_i) d\Pi(\theta)}{\int \prod_{i\in S_0^c}f_{\theta_i}(y_i) d\Pi(\theta)} \\
&=\frac{\int_{ \mathcal B_n} \prod_{i \in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)} d\Pi(\theta)}{\int \prod_{i\in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)} d\Pi(\theta)} \\
&\stackrel{let}{=}\frac{\mathcal N^\prime_n}{\mathcal D^\prime_n} \\
\end{align*}
$$

이에 대해 사건 $\mathcal A^\prime_n$을 다음과 같이 정의한다. 이 때의 $r_n$은 $r_n^2 = q_n$로 정의한다.

$$
\begin{align*}
\mathcal A_n^\prime &= \{ \mathcal D_n^\prime \geq e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c} - \mathbf 0\Vert_2 \leq 2r_n)\} \\
&= \{ \mathcal D_n^\prime \geq e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)\} 
\end{align*}
$$

이를 이용하면, 

$$
\begin{align*}
E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)}) &= E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})I_{\mathcal A^\prime_n} + E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})I_{\mathcal A^{\prime c}_n} \\
&\leq E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})I_{\mathcal A^\prime_n} + E_{\theta_0}I_{\mathcal A^{\prime c}_n} \\
&= E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})I_{\mathcal A^\prime_n} + P_{\theta_0}(\mathcal A^{\prime c}_n) \\
\end{align*}
$$

$\theta = \theta_{S_0^c}, \theta_0 = \mathbf 0$로 두고 **1)**에서 증명한 결과를 사용하면 다음을 얻을 수 있다. 사실 $\theta_0$는 $S_0^c$의 entry에서는 $0$의 값을 가지므로, $\theta_0 = \mathbf 0 = \theta_{0S_0^c}$이다.

$$
\begin{align*}
E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)}) &\leq E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})I_{\mathcal A^\prime_n} + P_{\theta_0}(\mathcal A^{\prime c}_n) \\
&\leq E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})I_{\mathcal A^\prime_n} + e^{-r_n^2/2} \\
&= E_{\theta_0}\left[\frac{\mathcal N^\prime_n}{\mathcal D^\prime_n}I_{\mathcal A^\prime_n} \right]+ e^{-r_n^2/2} \\
&\leq E_{\theta_0}\frac{\int_{ \mathcal B_n} \prod_{i \in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)} d\Pi(\theta)}{e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)} + e^{-r_n^2/2}
\end{align*}
$$

우변 첫 번째 항의 분자를 정리하면 다음과 같다.

$$
\begin{align*}
E_{\theta_0}\int_{ \mathcal B_n} \prod_{i \in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)} d\Pi(\theta) &= \iint_{ \mathcal B_n} \prod_{i \in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)}\left[\prod_{i=1}^{n}{f_{\theta_{0i}}(y_i)}\right] d\Pi(\theta)dy\\
&= \iint_{ \mathcal B_n} \prod_{i \in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)}\left[\prod_{i \in S_0^c}{f_{\theta_{0i}}(y_i)}\right] d\Pi(\theta)dy\\
&= \iint_{ \mathcal B_n} \prod_{i \in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)}\left[\prod_{i \in S_0^c}{f_{0}(y_i)}\right] d\Pi(\theta)dy\\
&= \iint_{ \mathcal B_n} \prod_{i \in S_0^c}f_{\theta_i}(y_i) d\Pi(\theta)dy\\
&= \int_{ \mathcal B_n}\int \prod_{i \in S_0^c}f_{\theta_i}(y_i) dyd\Pi(\theta)\\
&= \int_{ \mathcal B_n}d\Pi(\theta)\\
&= \Pi(\mathcal B_n)\\
\end{align*}
$$

다섯 번째 등호는 Fubini 정리에 의해 성립한다. 이 결과를 위 부등식에 대입하면 다음과 같다.

$$
\begin{align*}
E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)}) &\leq E_{\theta_0}\frac{\int_{ \mathcal B_n} \prod_{i \in S_0^c}\frac{f_{\theta_i}(y_i)}{f_0(y_i)} d\Pi(\theta)}{e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)} + e^{-r_n^2/2} \\
&= \frac{\Pi(\mathcal B_n)}{e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)} + e^{-r_n^2/2} \\
&= \frac{\Pi(\vert \text{supp}_{\delta_n}(\theta) \cap S_0^c\vert > A q_n)}{e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)} + e^{-r_n^2/2} \\
&\leq \frac{\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n)}{e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)} + e^{-r_n^2/2} \\
\end{align*}
$$

Cauchy-Schwartz 부등식을 사용하면, $\Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)$는 다음을 만족한다.

$$
\begin{align*}
\Vert \theta_{S_0^c}\Vert_2 = \sqrt{\sum_{j \in S_0^c}\vert \theta_j\vert^2} &\leq \sum_{j \in S_0^c}\vert \theta_j\vert \\
\implies \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n) &\geq \Pi\left(\sum_{j \in S_0^c}\vert \theta_j\vert\leq 2r_n\right) \\
&\geq \Pi\left(\vert \theta_j\vert\leq \frac{2r_n}{\sqrt{\vert S_0^c\vert}}, \text{ }\forall j \in S_0^c\right) \\
&\geq \Pi\left(\vert \theta_j\vert\leq \frac{2r_n}{\sqrt n}, \text{ }\forall j \in S_0^c\right) \\
&= \Pi\left(\vert \theta_1\vert\leq \frac{2r_n}{\sqrt n}\right)^{n-q_n}
\end{align*}
$$

**5.1.2)**에서 증명한 lemma를 사용하면 어떤 상수 $C$에 대해 다음이 만족한다.

$$
\begin{align*}
\Pi\left(\vert \theta_1\vert>\frac{2r_n}{\sqrt n}\right) &\leq \frac{C}{\Gamma(a_n)}\left(\frac{1}{2}\log n - \log 2 - \frac{1}{2}\log q_n -\frac{1}{2}\log \log n\right) \\
&\leq  \frac{C}{n^{1+\beta}}\left(\frac{1}{2}\log n - \log 2 - \frac{1}{2}\log q_n -\frac{1}{2}\log \log n\right) \\
&\leq  C\frac{\log n}{n^{1+\beta}} \\
&\leq  C\frac{\log n}{n} \\
\Pi\left(\vert \theta_1\vert \leq \frac{2r_n}{\sqrt n}\right)^{n-q_n} &\geq \left( 1- C\frac{\log n}{n}\right)^{n-q_n} \\
&> \left( 1- C\frac{\log n}{n}\right)^{n} \\
&\asymp e^{-C\log n}
\end{align*}
$$

두 번째 부등호는 $x$가 작을 때 $\Gamma(x) \leq 1/x$인 사실을 이용한 것이다. 세 번째 부등호에서 알 수 있듯, 이 부분은 $a_n = 1/n$, $a_n =1/n^{1+\beta}$ 두 경우에서 모두 만족한다. 이에 따라 충분히 큰 $n$과 어떤 상수 $n$에 대해서 다음이 만족한다.

$$
\Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n) \geq e^{-C\log n}
$$


이제 분자의 $\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n)$를 bound시키자. 먼저 $a_n = 1/n $인 경우를 고려한다. $\vert \text{supp}_{\delta_n}(\theta)\vert$는 $\zeta_n = \Pi(\vert \theta_1\vert > \delta_n)$에 대하여 $\text{Binom}(n, \zeta_n)$을 따르는 확률변수로 볼 수 있다. 이 때 $\zeta_n$은 바로 위에서와 같이 **5.1.2)**의 lemma에 의해 다음이 만족한다.

$$
\zeta_n = \Pi(\vert \theta_1\vert > \delta_n) \leq C^\prime \frac{\log n}{n}
$$

이항분포에 대한 Chernoff 부등식은 다음과 같다.  

*For $B \sim \text{Binom}(n, \zeta)$ and $\zeta \leq b < 1$,* 

$$
\mathbb P(B > bn) \leq \left\{ \left( \frac{\zeta}{b}\right)^b e^{b-\zeta}\right\}^n
$$

$a_n = 1/n $인 경우 $q_n \gtrsim \log n$을 추가로 가정하기 때문에, 적절한 상수 $C_0$에 대해 $q_n \geq C_0 \log n$이 만족한다. $A > C^\prime/C_0$인 상수 $A$에 대해 $b_n = Aq_n /n$로 설정한다. 이 때 다음은 자동으로 만족한다.

$$
b_n = \frac{Aq_n}{n} > \frac{C^\prime q_n}{C_0n} \geq \frac{C^\prime \log n}{n} \geq \zeta_n
$$

Chernoff 부등식을 적용하고 위 부등식 관계를 대입하면 다음과 같다.

$$
\begin{align*}
\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n) &= \Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > b_n n)\\
&\leq \left\{ \left( \frac{\zeta_n}{Aq_n /n}\right)^{Aq_n /n} e^{Aq_n /n-\zeta_n}\right\}^n\\
&= \left( \frac{n\zeta_n}{Aq_n}\right)^{Aq_n} e^{Aq_n-n\zeta_n} \\
&\leq \left( \frac{C^\prime \log n}{Aq_n}\right)^{Aq_n} e^{Aq_n} \\
&= e^{Aq_n \log(eC^\prime \log n) - A q_n \log(Aq_n)} \\
&= e^{-Aq_n \log\left({Aq_n}/{eC^\prime \log n}\right)} \\
&\leq e^{-Aq_n \log\left(q_n/eC_0 \log n\right)} \\
&\leq e^{-Aq_n M} \\
\end{align*}
$$

$q_n/\log n$이 bounded 혹은 $\infty$로 발산하는 수열이므로 마지막 부등호가 성립하도록 적절하게 상수 $M$을 잡아줄 수 있다. 즉 $q_n \gtrsim \log n$ 하에서 $\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n)$는 0으로 수렴하게 된다.



$a_n = 1/n^{1+\beta}$인 경우, **5.1.2)**의 lemma에 의해 다음이 성립한다.

$$
\zeta_n = \Pi(\vert \theta_1\vert > \delta_n) \leq C^\prime \frac{\log n}{n^{1+\beta}}
$$

$b_n$은 위에서와 마찬가지로 $b_n = Aq_n /n$으로 설정한다. 이 때는 $q_n \gtrsim \log n$을 가정하지 않아도, 충분히 큰 $n$에 대해서는 다음과 같이 $\zeta_n \leq b_n$이 성립한다.

$$
\begin{align*}
n\zeta_n \leq \frac{C^\prime \log n}{n^\beta} \rightarrow 0, \text{ as }n \rightarrow \infty. &\implies n\zeta_n \leq Aq_n \\
&\implies \zeta_n\leq \frac{Aq_n}{n} = b_n
\end{align*}
$$

Chernoff 부등식을 적용하고 위 부등식 관계를 대입하면 다음과 같다.

$$
\begin{align*}
\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n) &= \Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > b_n n)\\
&\leq \left\{ \left( \frac{\zeta_n}{Aq_n /n}\right)^{Aq_n /n} e^{Aq_n /n-\zeta_n}\right\}^n\\
&= \left( \frac{n\zeta_n}{Aq_n}\right)^{Aq_n} e^{Aq_n-n\zeta_n} \\
&\leq \left( \frac{C^\prime \log n}{Aq_n n^\beta}\right)^{Aq_n} e^{Aq_n} \\
&= e^{Aq_n \log(eC^\prime \log n) - A q_n \log(Aq_n n^\beta)} \\
&= e^{-Aq_n \log\left({Aq_n n^\beta}/{eC^\prime \log n}\right)} \\
&\leq e^{-Aq_n \log\left(n^\beta/e\right)} \\
&= e^{-Aq_n \{\beta\log n -1\}} \\
\end{align*}
$$

즉 $a_n = 1/n^{1+\beta}$의 경우에도, $\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n)$는 0으로 수렴한다.



따라서, 다시 원래의 부등식으로 돌아가서 도출한 결과들을 대입하면 다음과 같다. 먼저 $a_n = 1/n$의 경우에는 다음과 같다.

$$
\begin{align*}
E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})&\leq \frac{\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n)}{e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)} + e^{-r_n^2/2} \\
&\leq \frac{e^{-Aq_n M}}{e^{-4r_n^2} e^{-C\log n}} + e^{-r_n^2/2} \\
\end{align*}
$$

$r_n^2 = q_n\log n$으로 설정했던 다른 파트들과 표현이 다소 헷갈리지만, 이 **3)**의 증명에서는 $r_n^2 = q_n$로 설정하였다. 따라서 $A$를 충분히 크게 잡으면, $q_n \gtrsim \log n$이 가정되었으므로, 위 식이 0으로 수렴하도록 할 수 있다.  

$a_n = 1/n^{1+\beta}$의 경우는 다음과 같다.

$$
\begin{align*}
E_{\theta_0}\Pi_n( \mathcal B_n \vert y^{(n)})&\leq \frac{\Pi(\vert \text{supp}_{\delta_n}(\theta)\vert > A q_n)}{e^{-4r_n^2} \Pi(\Vert \theta_{S_0^c}\Vert_2 \leq 2r_n)} + e^{-r_n^2/2} \\
&\leq \frac{e^{-Aq_n \{\beta\log n -1\}}}{e^{-4r_n^2} e^{-C\log n}} + e^{-r_n^2/2} \\
\end{align*}
$$

이 경우 역시 $A$를 충분히 크게 잡으면, 위 식이 0으로 수렴하도록 할 수 있다. 이것으로 **3)**이 증명되었다.





## 4) Construction of test

이 부분은 Castillo and van der Vaart (2012)의 **Lemma 5.1**이다. Statement는 다음과 같다.

*For any $\alpha, \beta >0$, and any $\theta_0, \theta_1 \in \mathbb R^n$, there exists a test $\phi$ based on $y \sim \text{N}(\theta, I)$, such that for every $\theta \in \mathbb R^n$ with $\Vert \theta - \theta_1 \Vert \leq \Vert \theta_0 - \theta_1 \Vert/2 \stackrel{\triangle}{=} \rho$,*

$$
\begin{align*}
\alpha E_{\theta_0}\phi + \beta E_\theta(1-\phi) &\leq \alpha \left[ 1- \Phi\Big(\frac{\rho}{2} + \frac{1}{\rho} \log\frac{\alpha}{\beta} \Big)\right] + \beta \Phi\Big(-\frac{\rho}{2} + \frac{1}{\rho} \log\frac{\alpha}{\beta} \Big) \\
&\leq 2\sqrt{\alpha \beta}e^{-\Vert \theta_0 - \theta_1 \Vert^2 /32}
\end{align*}
$$

WLOG, $\theta_0 = \mathbf 0$을 가정하고 위 사실을 증명해도 충분하다.  

$\Vert \theta - \theta_1 \Vert \leq \Vert \theta_1 \Vert/2 = \rho$이면, $\Vert \theta \Vert \geq \Vert \theta_1  \Vert/2$이므로, ($\theta, \theta_1$의 그림을 그려보면 자명함)

$$
\begin{align*}
\langle \theta_1, \theta\rangle &= \langle \theta_1 - \theta , \theta\rangle + \langle  \theta, \theta\rangle \\
 &= \langle \theta_1 - \theta , \theta - \theta_1 \rangle + \langle \theta_1 - \theta , \theta_1 \rangle + \langle  \theta, \theta\rangle \\
 &= - \Vert \theta_1 - \theta\Vert^2 + \Vert \theta_1 \Vert^2 + \Vert \theta \Vert^2 - \langle \theta , \theta_1 \rangle \\
\langle \theta_1, \theta\rangle  &= \frac{1}{2}(\Vert \theta_1 \Vert^2 +\Vert \theta \Vert^2 - \Vert \theta_1 - \theta\Vert^2 ) \\
&\geq \frac{1}{2}\Vert \theta_1 \Vert^2 \\
&\because \enspace \Vert \theta \Vert \geq \Vert \theta - \theta_1 \Vert 
\end{align*}
$$

$\phi(y) = 1_{ \theta_1^Ty > D \Vert \theta_1 \Vert}$로 두면, $y$는 정규분포를 따르므로 다음이 만족한다.

$$
\begin{align*}
E_{\theta_0}\phi(y) &= 1-\Phi\left( D\right) \\
E_{\theta}(1-\phi(y)) &= \Phi\left( D - \frac{\theta_1^T \theta}{\Vert \theta_1 \Vert}\right) \\
&\leq \Phi\left( D - \frac{\Vert \theta_1 \Vert}{2}\right) =\Phi( D - \rho) 
\end{align*}
$$

마지막 줄의 부등호는 위에서 보인 $\langle \theta_1, \theta\rangle \geq \frac{1}{2}\Vert \theta_1 \Vert^2$에 의한 것이다. 위 사실들을 종합하면,

$$
\begin{align*}
\alpha E_{\theta_0}\phi + \beta E_\theta (1-\phi) &\leq \alpha \Big[ 1-\Phi\left( D\right)\Big]  + \beta\Phi( D - \rho)
\end{align*}
$$

이는 임의의 $D$에 대해 성립하므로, $\alpha [ 1-\Phi\left( D\right)]  + 
\beta\Phi( D - \rho)$를 minimize하는 $D = \frac{1}{\rho}\log \frac{\alpha}{\beta} + \frac{1}{2}\rho$에 대해 위 부등식을 나타내면 statement의 첫 번째 부등식을 보일 수 있다.

$$
\alpha E_{\theta_0}\phi + \beta E_\theta (1-\phi) \leq \alpha \Big[ 1-\Phi\left( \frac{1}{\rho}\log \frac{\alpha}{\beta} + \frac{1}{2}\rho\right)\Big]  + 
\beta\Phi\left( \frac{1}{\rho}\log \frac{\alpha}{\beta} - \frac{1}{2}\rho \right)
$$

두 번째 부등식은 (i) $D - \rho \leq 0 \leq D$, (ii) $ 0 < D-\rho < D$, (iii) $D- \rho \leq D < 0$의 세 경우로 나누어 보일 수 있다.

(i) $D - \rho \leq 0 \leq D$의 경우, $x > 0 \implies 1-\Phi(x) \leq e^{-x^2/2}$를 이용하여 식을 잘 정리하면 다음과 같다.

$$
\begin{align*}
\alpha E_{\theta_0}\phi + \beta E_\theta (1-\phi) &\leq \alpha \Big[ 1-\Phi\left( D\right)\Big]  + \beta\Phi( D - \rho) \\ 
&\leq \alpha e^{-D^2/2} + \beta e^{-(D-\rho)^2/2} \\ 
&\leq \left[ \alpha\left(\frac{\alpha}{\beta}\right)^{-1/2} + \beta\left(\frac{\alpha}{\beta}\right)^{1/2} \right]e^{- \left( \log \frac{\alpha}{\beta}\right)^2/2\rho^2}e^{-{\rho^2}/{8}} \\ 
&\leq 2 \sqrt{\alpha\beta} \text{ }e^{- \left( \log \frac{\alpha}{\beta}\right)^2/2\rho^2}e^{-{\rho^2}/{8}} \\ 
\end{align*}
$$

(ii) $ 0 < D-\rho < D$의 경우, $\alpha [ 1-\Phi\left( D\right)] $는 (i)와 동일하게 bound를 잡을 수 있다. 

$$
\alpha \Big[ 1-\Phi\left( D\right)\Big] \leq \sqrt{\alpha \beta} e^{- \left( \log \frac{\alpha}{\beta}\right)^2/2\rho^2}e^{-{\rho^2}/{8}}
$$

또한 $D-\rho > 0 $이므로,

$$
\begin{align*}
D-&\rho = \frac{1}{\rho}\log \frac{\alpha}{\beta} - \frac{1}{2}\rho > 0 \\
\implies \beta &< \alpha e^{-\rho^2/2}
\end{align*}
$$

이를 이용하면, 다음과 같이 $\beta \Phi( D - \rho)$를 bound시킬 수 있다.

$$
\begin{align*}
\beta \Phi( D - \rho)&\leq \beta \\
&= \sqrt \beta \sqrt \beta \\
&\leq \sqrt \beta \sqrt{\alpha e^{-\rho^2/2}}\\
&=\sqrt{\alpha \beta}e^{-\rho^2/4} \\
&\leq \sqrt{\alpha \beta}e^{-\rho^2/8}
\end{align*}
$$

(iii) $D- \rho \leq D < 0$의 경우, $\beta\Phi( D - \rho)$는 (i)와 동일하게 bound를 잡을 수 있다.

$$
\beta\Phi( D - \rho) \leq \sqrt{\alpha \beta} e^{- \left( \log \frac{\alpha}{\beta}\right)^2/2\rho^2}e^{-{\rho^2}/{8}}
$$

 $D < 0 $이므로,
 
$$
\begin{align*}
D& = \frac{1}{\rho}\log \frac{\alpha}{\beta} + \frac{1}{2}\rho < 0 \\
\implies \alpha &< \beta e^{-\rho^2/2}
\end{align*}
$$

이를 이용하면, 다음과 같이 $\alpha [ 1-\Phi\left( D\right)] $를 bound시킬 수 있다.

$$
\begin{align*}
\alpha \Big[ 1- \Phi( D )\Big]&\leq  \alpha \\
&= \sqrt{\alpha}\sqrt{\alpha} \\
&\leq \sqrt{\alpha}\sqrt{\beta e^{-\rho^2/2}}\\
&= \sqrt{\alpha \beta}e^{-\rho^2/4} \\
&\leq \sqrt{\alpha \beta}e^{-\rho^2/8}
\end{align*}
$$

다음 결과를 확인하자.

$$
- \frac{1}{2 \rho^2}\left( \log \frac{\alpha}{\beta}\right)^2 < 0\\
\implies e^{- \left( \log \frac{\alpha}{\beta}\right)^2/2\rho^2} < 1
$$

따라서 (i), (ii), (iii)의 모든 경우에서 다음 bound가 성립하고, $\Vert \theta_1 \Vert/2 = \rho$를 이용하면 증명이 끝난다.

$$
\begin{align*}
\therefore \alpha E_{\theta_0}\phi + \beta E_\theta (1-\phi) &\leq 2 \sqrt{\alpha\beta} \text{ }e^{-{\rho^2}/{8}} \\
&= 2 \sqrt{\alpha\beta} \text{ }e^{-{\Vert \theta_1 \Vert^2}/{32}} \\ 
\end{align*}
$$



## 5) Upper bound of $\beta_{S,j,i}$

$\beta_{S,j,i}$는 다음과 같이 정의된다.

$$
\beta_{S,j,i}= \frac{\Pi(B_{S,j,i})}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
$$

이 때 $B_{S,j,i}$는 2)에서 정의된 $\theta^{S,j,i}$를 중심으로 갖는 $S$ 위의 반지름 $jr_n$의 ball이다.

$$
B_{S,j,i} = \Big\{ \theta : \text{supp}_{\delta_n}(\theta) = S ,\Vert \theta - \theta^{S,j,i} \Vert < jr_n\Big\}
$$

이 upper bound는 논문의 **Lemma 6.1**에 소개된 bound이다.

$$
\log \beta_{S,j,i} \leq \vert S \vert \log j  + C(\vert S \vert+\vert S_0 \vert) \log n + C^\prime r_n^2
$$



### 5.1) Necessary lemmas

다음 lemma들은 minimax rate 증명과정에 필요한 $\text{DL}_{a_n}$ prior의 몇 가지 부등식이다.



### 5.1.1) Bounds of joint prior density of $\text{DL}_a $

이 lemma는 논문의 **Lemma 3.1**에 해당한다. Statement는 다음과 같다.

*Consider the $\text{DL}_a$ prior on $\mathbb R^n$ for $a$ small. Let $S \subset \{ 1, \ldots , n\}$ and $\eta \in \mathbb R^{\vert S \vert}$. If $\min_{1\leq j \leq \vert S \vert } \vert \eta_j \vert > \delta$ for $\delta $ small, then*

$$
\log \Pi_S(\eta) \leq C \vert S \vert \log(1/\delta)
$$

*where $C>0$ is an absolute constant.*

*If $\Vert \eta \Vert_2 \leq m $ for $m$ large, then*

$$
\log \Pi_S(\eta) \geq -C\{ \vert S\vert \log(1/a) + \vert S\vert^{3/4}m^{1/2}  \}
$$

*where $C>0$ is an absolute constant.*



### 5.1.2) Marginal prior density and lower bound of probability near zero

이 lemma는 논문의 **Proposition 3.1**과 **Lemma 3.2**에 해당한다. Statement는 다음과 같다.

*The marginal density $\Pi$ of $\theta_j$ for any $1\leq j \leq n$ is given by*

$$
\Pi(\theta_j) = \frac{1}{2^{(1+a)/2}\Gamma(a)} \vert \theta_j\vert^{(a-1)/2}K_{1-a}(\sqrt{2 \vert \theta_j \vert }) 
$$

*where* 

$$
K_\nu(x) = \frac{\Gamma(\nu+1/2)(2x)^\nu}{\sqrt{\pi}} \int_0^\infty \frac{\cos t }{(t^2 + x^2)^{(\nu+1)/2}}dt
$$

*is the modified Bessel function of the second kind.*

*Assume $\theta_1 \in \mathbb R$ has a probability density as above. Then, for $\delta>0$ small,*

$$
\Pi(\vert \theta_1 \vert > \delta) \leq C \log (1/\delta)/\Gamma(a)
$$

*where $C>0$ is an absolute constant.*



### 5.2) Proof

$B_{S,j,i}$의 정의를 활용하여 $\beta_{S,j,i}$의 분자를 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
\Pi(B_{S,j,i}) &= \Pi(\theta : \text{supp}_{\delta_n}(\theta) = S ,\Vert \theta - \theta^{S,j,i} \Vert < jr_n) \\
&\leq \Pi(\theta : \text{supp}_{\delta_n}(\theta) = S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n) \\
&= \Pi(\theta : \vert \theta_j \vert < \delta_n \text{ }\forall j\in S^c, \vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n) \\
&\stackrel{\text{iid}}{=} \Pi_{S^c}(\theta \in \mathbb R^{n-\vert S \vert}: \vert \theta_j \vert < \delta_n \text{ }\forall j\in S^c) \cdot \Pi_S(\theta \in \mathbb R^{\vert S \vert} :\vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n) \\
&\stackrel{\text{iid}}{=} \Pi_1(\theta_1 \in \mathbb R: \vert \theta_1 \vert < \delta_n)^{n-\vert S \vert} \cdot \Pi_S(\theta \in \mathbb R^{\vert S \vert} :\vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n)
\end{align*}
$$

이 때 $\Pi_1$는 $\theta_1$에 대한 prior distribution이며, 두 번째 줄의 부등호는 다음 성질에 의해 성립한다.

$$
\Vert \theta - \theta^{S,j,i}\Vert^2 = \Vert \theta_S - \theta_S^{S,j,i}\Vert^2 + \Vert \theta_{S^c} \Vert^2 \geq \Vert \theta_S - \theta_S^{S,j,i}\Vert^2 \\
\implies \Vert \theta - \theta^{S,j,i}\Vert\geq \Vert \theta_S - \theta_S^{S,j,i}\Vert \\
\implies \{ \theta: \Vert \theta - \theta^{S,j,i}\Vert < jr_n\} \subset \{ \theta:\Vert \theta_S - \theta_S^{S,j,i}\Vert< jr_n \} \\
$$

$\beta_{S,j,i}$의 분모의 $\Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)$는 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
\Pi(\Vert \theta - \theta_0\Vert \leq 2r_n) &\geq \Pi(\Vert \theta_{S_0} - \theta_{0S_0}\Vert \leq r_n, \Vert \theta_{S_0^c}\Vert \leq r_n) \\
&\stackrel{\text{iid}}{=}\Pi_{S_0^c}( \theta \in \mathbb R^{n-q_n}:\Vert \theta_{S_0^c}\Vert \leq r_n)\Pi_{S_0}(\theta \in \mathbb R^{q_n}:\Vert \theta_{S_0} - \theta_{0S_0}\Vert \leq r_n) \\
&\stackrel{\text{iid}}{=}\Pi_1( \theta_1 \in \mathbb R:\vert \theta_1\vert \leq r_n)^{n-q_n}\Pi_{S_0}(\theta \in \mathbb R^{q_n}:\Vert \theta_{S_0} - \theta_{0S_0}\Vert_2 \leq r_n)
\end{align*}
$$

이를 종합하면 $\beta_{S,j,i}$를 다음과 같이 bound시킬 수 있다.

$$
\begin{align*}
\beta_{S,j,i} &\leq \frac{e^{4r_n^2}\Pi_1(\vert \theta_1 \vert < \delta_n)^{n-\vert S \vert} \Pi_S(\vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n)}{ \Pi_1(\vert \theta_1\vert \leq r_n)^{n-q_n}\Pi_{S_0}(\Vert \theta_{S_0} - \theta_{0S_0}\Vert_2 \leq r_n)} \\
&= \frac{e^{4r_n^2} }{ \Pi_1(\vert \theta_1\vert \leq r_n)^{\vert S \vert -q_n}} R_{S,j,i} \\
&\leq\frac{e^{4r_n^2} }{ \Pi_1(\vert \theta_1\vert \leq r_n)^{\vert S \vert }} R_{S,j,i}
\end{align*}
$$

$ R_{S,j,i} = \frac{ \Pi_S(\vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n)}{\Pi_{S_0}(\Vert \theta_{S_0} - \theta_{0S_0}\Vert_2 \leq r_n)}$의 분자와 분모의 bound를 구해보자. 먼저 $R_{S,j,i}$의 분자는 다음과 같이 bound될 수 있다.

$$
\Pi_S(\vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n) \leq \left[ \sup_{\vert \theta_j \vert > \delta_n \text{ }\forall j\in S}\Pi_S(\theta_S) \right] \left\vert v_{\vert S \vert}(jr_n)\right\vert
$$

위 부등식 관계는 집합 $\{\theta_S \in \mathbb R^{\vert S\vert}:\vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n\}$에서 uniform하게 $\sup_{\vert \theta_j \vert > \delta_n \text{ }\forall j\in S}\Pi_S(\theta_S)$의 mass를 주는 measure로 $\Pi_S$를 bound시킨 것으로 이해할 수 있다. 이 때 $v_q(r)$은 $\mathbb R^q$ 내의 center $\mathbf 0$, 반지름 $r$의 ball을 나타내며, $\vert v_q(r) \vert$은 그 부피를 나타낸다.

$R_{S,j,i}$의 분모는 다음과 같이 나타낼 수 있다.

$$
\begin{align*}
\Pi_{S_0}(\Vert \theta_{S_0} - \theta_{0S_0}\Vert_2 \leq r_n) &\geq \left[ \inf_{\theta_{S_0}:\Vert\theta_{S_0}  - \theta_{0S_0}\Vert < r_n}\Pi_{S_0}(\theta_{S_0}) \right] \left\vert v_{\vert S_0 \vert}(r_n) \right\vert\\
&\geq \left[ \inf_{\theta_{S_0} \in v_{\vert S_0\vert} (t_n)}\Pi_{S_0}(\theta_{S_0}) \right] \left\vert v_{\vert S_0 \vert}(r_n) \right\vert
\end{align*}
$$

이 때 $t_n = \Vert \theta_{0S_0}\Vert + r_n = \Vert \theta_{0}\Vert + r_n$이며, 두 번째 부등식은 다음 부분집합 관계에 의해 성립한다.

$$
\{ \theta_{S_0}:\Vert\theta_{S_0}  - \theta_{0S_0}\Vert < r_n\}\subset v_{\vert S_0\vert} (t_n) = \{ \theta_{S_0}:\Vert\theta_{S_0}  \Vert < \Vert \theta_{0S_0}\Vert + r_n \}
$$

두 부등식의 $\sup$, $\inf$ 부분에 대해 **5.1.1)**의 lemma를 사용하면 다음과 같다.

$$
\sup_{\vert \theta_j \vert > \delta_n \text{ }\forall j\in S}\Pi_S(\theta_S) \leq e^{C\vert S\vert \log(1/\delta_n)} \\
\inf_{\theta_{S_0} \in v_{\vert S_0\vert} (t_n)}\Pi_{S_0}(\theta_{S_0}) \geq e^{-C\left\{\vert S\vert \log(1/a_n) + \vert S \vert^{3/4}t_n^{1/2} \right\}}
$$

이 결과를 적용하면 다음과 같다.

$$
\begin{align*}
R_{S,j,i} &= \frac{ \Pi_S(\vert \theta_j \vert > \delta_n \text{ }\forall j\in S ,\Vert \theta_S - \theta_S^{S,j,i} \Vert < jr_n)}{\Pi_{S_0}(\Vert \theta_{S_0} - \theta_{0S_0}\Vert_2 \leq r_n)} \\
&\leq \frac{ \left[ \sup_{\vert \theta_j \vert > \delta_n ,\forall j\in S}\Pi_S(\theta_S) \right] \left\vert v_{\vert S \vert}(jr_n)\right\vert}{ \left[ \inf_{\theta_{S_0} \in v_{\vert S_0\vert} (t_n)}\Pi_{S_0}(\theta_{S_0}) \right] \left\vert v_{\vert S_0 \vert}(r_n) \right\vert} \\
&\leq \frac{ \left\vert v_{\vert S \vert}(jr_n)\right\vert \exp({C\vert S\vert \log(1/\delta_n)})  }{ \left\vert v_{\vert S_0 \vert}(r_n) \right\vert \exp({-C\left\{\vert S_0\vert \log(1/a_n) + \vert S_0 \vert^{3/4}t_n^{1/2} \right\}}) } \\
&\leq \frac{ j^{\vert S\vert }r_n^{\vert S\vert }\left\vert v_{\vert S \vert}(1)\right\vert \exp({C\vert S\vert \log(1/\delta_n)})  }{ r_n^{\vert S_0 \vert}\left\vert v_{\vert S_0 \vert}(1) \right\vert \exp({-C\left\{\vert S_0\vert \log(1/a_n) + \vert S_0 \vert^{3/4}t_n^{1/2} \right\}}) }
\end{align*}
$$

네 번째 줄의 부등호는 $\mathbb R^q$ 내의 center $\mathbf 0$, 반지름 $r$의 ball $v_q(r)$의 부피를 $r^q \vert v_q(1)\vert$로 나타낸 것이다. 양변에 로그를 취하면,

$$
\begin{align*}
\log R_{S,j,i} &\leq \vert S \vert \log j + \log\frac{r_n^{\vert S\vert }\left\vert v_{\vert S \vert}(1)\right\vert}{r_n^{\vert S_0 \vert}\left\vert v_{\vert S_0 \vert}(1) \right\vert}  + C \left\{ \vert S\vert\log(1/\delta_n) + \vert S_0\vert \log(1/a_n) + \vert S_0 \vert^{3/4}t_n^{1/2} \right\} \\
&\leq \vert S \vert \log j  + C\left\{ \vert S \vert \log n + r_n^2+\vert S\vert \log(1/\delta_n) + \vert S_0\vert \log(1/a_n) + \vert S_0 \vert^{3/4}t_n^{1/2} \right\} 
\end{align*}
$$

두 번째 부등호는 다음 결과를 사용한 것이다. 감마함수의 Stirling 근사와 그 bound를 통해 unit ball의 volume에 대한 bound를 구하고, 이를 대입하여 정리하면 다음과 같다. 

$$
\begin{align*}
&\sqrt{2\pi x} \left(\frac{x}{e} \right)^x<\Gamma(x+1) < \sqrt{2\pi x} \left(\frac{x}{e} \right)^x e^{1/12x}\\
&\implies \pi^{-1/2}e^{-1/6q}(2e\pi)^{q/2}q^{-(q+1)/2} < \left\vert v_q(1)\right\vert = \frac{\pi^{q/2}}{\Gamma(\frac{q}{2} + 1)} < \pi^{-1/2}(2e\pi)^{q/2}q^{-(q+1)/2} \\
&\implies \frac{\left\vert v_{\vert S \vert}(1)\right\vert}{\left\vert v_{\vert S_0 \vert}(1)\right\vert} < \frac{(2e\pi)^{\vert S\vert /2}\vert S\vert^{-(\vert S\vert+1)/2}}{(2e\pi)^{\vert S_0\vert/2}\vert S_0\vert^{-(\vert S_0\vert+1)/2}e^{-1/6}}
\end{align*}
$$

$$
\begin{align*}
&\log\frac{r_n^{\vert S\vert }\left\vert v_{\vert S \vert}(1)\right\vert}{r_n^{\vert S_0 \vert}\left\vert v_{\vert S_0 \vert}(1) \right\vert} \\
&\leq \left\{\frac{\vert S \vert}{2}-\frac{\vert S_0 \vert}{2} \right\} (\log r_n^2 + \log(2e\pi)) -\frac{\vert S\vert}{2}\log\vert S\vert +\frac{\vert S_0\vert}{2}\log\vert S_0\vert-\frac{1}{2}\log\frac{\vert S\vert}{\vert S_0\vert}+\frac{1}{6} \\
&= \left\{\frac{\vert S \vert}{2}-\frac{q_n}{2} \right\} (\log n + \log\log q_n + \log(2e\pi)) -\frac{(\vert S\vert + 1)}{2}\log\vert S\vert +\frac{q_n}{2}\log q_n+\frac{1}{2}\log q_n+\frac{1}{6} \\
&\leq \frac{\vert S \vert}{2} (\log n + \log\log q_n + \log(2e\pi)) +\frac{q_n}{2}\log q_n+\frac{1}{2}\log q_n+\frac{1}{6} \\
&\leq C \{ \vert S \vert \log n + q_n \log n \} \\
&= C \{ \vert S \vert \log n + r_n^2 \}
\end{align*}
$$

두 번째 줄의 등호는 $\vert S_0 \vert = q_n$를 사용한 것이며, 네 번째 줄의 부등호는 상수 $C$를 적절히 잡아 만족시킬 수 있다. 마지막 등호는 $r_n^2 = q_n \log n$을 사용한 것이다. 다시 $\log R_{S,j,i}$의 bound로 돌아가면, 다음과 같이 부등식을 나타낼 수 있다.

$$
\begin{align*}
\log R_{S,j,i} &\leq \vert S \vert \log j  + C\left\{ \vert S \vert \log n + r_n^2+\vert S\vert \log(1/\delta_n) + \vert S_0\vert \log(1/a_n) + \vert S_0 \vert^{3/4}t_n^{1/2} \right\} \\
&\leq \vert S \vert \log j  + C(1+\beta)(\vert S \vert+\vert S_0 \vert) \log n + C r_n^2 + C\vert S \vert \log(1/\delta_n) + \vert S_0 \vert^{3/4} t_n^{1/2}\\
&= \vert S \vert \log j  + C(\vert S \vert+(1 +\beta)\vert S_0 \vert) \log n + C r_n^2 + C\vert S \vert \log n - C\vert S \vert \log r_n + \vert S_0 \vert^{3/4} t_n^{1/2}\\
&= \vert S \vert \log j  + C(2\vert S \vert+(1 +\beta)\vert S_0 \vert) \log n + C r_n^2 - C\vert S \vert \log r_n + \vert S_0 \vert^{3/4} t_n^{1/2}\\
&\leq \vert S \vert \log j  + C(2\vee 1+\beta)(\vert S \vert+\vert S_0 \vert) \log n + C r_n^2 + q_n^{3/4} t_n^{1/2}\\
&\leq \vert S \vert \log j  + C(2 \vee 1+\beta)(\vert S \vert+\vert S_0 \vert) \log n + C^\prime r_n^2
\end{align*}
$$

세 번째 줄의 등호는 $\delta_n = \frac{r_n}{n}$을 사용한 것이고, 다섯 번째 줄의 부등호는 우변 전체의 속도에 영향을 주지 않는 moderate한 속도의 음의 항 $-C \vert S\vert \log r_n$을 날려준 것이다. 마지막 부등호는 다음과 같은 결과에 의해 적절히 상수 $C^\prime$을 잡아준 것이다. 여기서 맨 처음에 가정했던 $ \Vert \theta_0 \Vert \leq q_n \log^2 n$의 조건이 사용된다.

$$
\begin{align*}
q_n^{3/2} t_n &= q_n^{3/2}(q_n^{1/2} \log^{1/2} n + \Vert \theta_0 \Vert) \\
&\leq q_n^{3/2}(q_n^{1/2} \log^{1/2} n + q_n^{1/2}\log^2n) \\
&\asymp q_n^2 \log^2 n \\
q_n^{3/4} t_n^{1/2} &\lesssim q_n \log n = r_n^2
\end{align*}
$$

이와 같은 $\log R_{S,j,i}$의 upper bound를 이용해 $\log \beta_{S,j,i}$를 bound시키면 다음과 같다.

$$
\begin{align*}
\log \beta_{S,j,i} &\leq 4r_n^2 - \vert S \vert \log\Pi_1(\vert \theta_1\vert \leq r_n)+ \log R_{S,j,i} \\
&\leq - \vert S \vert \log\Pi_1(\vert \theta_1\vert \leq r_n) + \vert S \vert \log j  + C(2 \vee 1+\beta)(\vert S \vert+\vert S_0 \vert) \log n + (C^\prime+4) r_n^2 \\
&\leq  - \log n + \vert S \vert \log j  + C(2 \vee 1+\beta)(\vert S \vert+\vert S_0 \vert) \log n + (C^\prime+4) r_n^2 \\
&\leq \vert S \vert \log j  + C(2 \vee 1+\beta)(\vert S \vert+\vert S_0 \vert) \log n + (C^\prime+4) r_n^2 
\end{align*}
$$

세 번째 부등호는 **5.1.2)**의 lemma를 사용한 것이다. 따라서 상수를 $C \stackrel{let}{=} C(2 \vee 1+\beta)$, $C^\prime \stackrel{let}{=} C^\prime + 4$로 표시하면 다음과 같은 결과를 얻는다.

$$
\log \beta_{S,j,i} \leq \vert S \vert \log j  + C(\vert S \vert+\vert S_0 \vert) \log n + C^\prime r_n^2
$$



## 6) The proof of the main result

우리는 다음을 보이고자 한다.

$$
E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n \vert y^{(n)} )  \rightarrow 0, \quad \text{as }n \rightarrow \infty
$$

좌변을 정리하면 다음과 같다.

$$
\begin{align*}
&E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n \vert y^{(n)} ) \\
&= E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} ) \\
&\enspace + E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n , \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} ) \\
&= E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} ) \\
&\enspace+ E_{\theta_0}\Pi_n( \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} ) \tag{6}
\end{align*}
$$

3)에서 우변의 두 번째 항 $E_{\theta_0}\Pi_n( \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} )$이 0으로 수렴하는 것을 보였기 때문에, 다음을 보이면 증명이 끝난다.

$$
\text{WTS : } E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} ) \rightarrow 0, \quad \text{as }n \rightarrow \infty
$$


**1)**에서 정의한 사건 $\mathcal A_n$에 대해 다음과 같이 부등식을 나타낼 수 있다.

$$
\begin{align*}
&E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} ) \\
&\leq E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} + E_{\theta_0}I_{\mathcal A_n^c} \\
&\leq  E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} + e^{-{r_n^2}/2} \tag{7}
\end{align*}
$$

아직 $r_n$을 특정하지는 않았지만, $n \rightarrow \infty$이면 $e^{-{r_n^2}/2} \rightarrow 0$인 $r_n$으로 설정한다면 우리는 다음을 보이면 된다.

$$
\text{WTS : } E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \rightarrow 0, \quad \text{as }n \rightarrow \infty
$$

$\vert S \vert \leq Aq_n$인 집합 $S$들의 모임으로 $\mathcal S_n$을 정의했기 때문에, 다음과 같이 정리할 수 있다.

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&= E_{\theta_0}\left[ \sum_{S \in \mathcal S_n}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \text{supp}_{\delta_n}(\theta) = S \vert y^{(n)} )I_{\mathcal A_n}\right] \\
&= E_{\theta_0}\left[ \sum_{S \in \mathcal S_n}\sum_{j \geq M}\Pi_n( \Theta_{S,j,n} \vert y^{(n)} )I_{\mathcal A_n}\right] \\
&\leq E_{\theta_0}\left[ \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}\Pi_n( B_{S,j,i}\vert y^{(n)} )I_{\mathcal A_n}\right] \\
\end{align*}
$$

두 번째 등호는 위에서 정의한 $\Theta_{S,j,n}$을 사용한 것이며, 마지막 줄의 부등호는 위에서와 같이 construct한 $\Theta_{S,j,n}$의 covering $\{B_{S,j,i}: i=1, \ldots , N_{S,j}\}$를 사용한 것이다. 

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq E_{\theta_0}\left[ \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}\Pi_n( B_{S,j,i}\vert y^{(n)} )I_{\mathcal A_n}\right] \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}} E_{\theta_0} \left[ \left\{ \phi_{S,j,i}\Pi_n( B_{S,j,i}\vert y^{(n)} ) + (1-\phi_{S,j,i})\Pi_n( B_{S,j,i}\vert y^{(n)} )\right\}I_{\mathcal A_n}\right] \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})\Pi_n( B_{S,j,i}\vert y^{(n)} )I_{\mathcal A_n}\right]\right\}
\end{align*}
$$

Posterior probability의 정의를 사용하면 $\Pi_n( B_{S,j,i}\vert y^{(n)} )$은 다음과 같다.

$$
\Pi_n( B_{S,j,i}\vert y^{(n)} ) = \frac{\int_{B^{S,j,i}} \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}{\int \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}
$$

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B_{S,j,i}} \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}{\int \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}
I_{\mathcal A_n}\right]\right\} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B_{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{\int \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}
I_{\mathcal A_n}\right]\right\} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B_{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{\mathcal D_n}
I_{\mathcal A_n}\right]\right\} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B_{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\right]\right\} 
\end{align*}
$$

마지막 부등호는 사건 $\mathcal A_n$ 위에서 $\mathcal D_n \geq e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)$가 성립함을 이용한 것이다. 또한 다음과 같은 사실을 이용할 수 있다.

$$
\begin{align*}
&E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B_{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\right] \\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\iint_{B^{S,j,i}}(1-\phi_{S,j,i}) \left[\prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)}\right]\left[\prod_{i=1}^{n}{f_{\theta_{0i}}(y_i)}\right]d\Pi(\theta)dy\\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\iint_{B_{S,j,i}}(1-\phi_{S,j,i}) \left[\prod_{i=1}^{n}{f_{\theta_i}(y_i)}\right]d\Pi(\theta)dy\\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\int_{B_{S,j,i}}\int(1-\phi_{S,j,i}) \left[\prod_{i=1}^{n}{f_{\theta_i}(y_i)}\right]dyd\Pi(\theta)\\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\int_{B_{S,j,i}}E_\theta (1-\phi_{S,j,i}) d\Pi(\theta)\\
&\leq \frac{\Pi(B_{S,j,i})}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\sup_{\theta \in B_{S,j,i}} \Big\{ E_\theta (1-\phi_{S,j,i})\Big\}\\
\end{align*}
$$

이를 대입하면,

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ \frac{\Pi(B_{S,j,i})}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\sup_{\theta \in B^{S,j,i}} \Big\{ E_\theta (1-\phi_{S,j,i})\Big\}\right\} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ \beta_{S,j,i}
\sup_{\theta \in B^{S,j,i}} \Big\{ E_\theta (1-\phi_{S,j,i})\Big\}\right\} \\
\end{align*}
$$

이 때 $ \beta_{S,j,i}= \frac{\Pi(B_{S,j,i})}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}$이다. 

$\theta_0 = \theta_0, \theta_1 = \theta^{S,j,i}$로 두면, 다음과 같이 **4)**의 조건이 만족하는 것을 확인할 수 있다.

$$
\theta \in B_{S,j,i} \implies \Vert \theta - \theta^{S,j,i}\Vert \leq jr_n \\
 \theta^{S,j,i} \in \Theta_{S,j,n} \implies  jr_n \leq \frac{1}{2}\Vert \theta^{S,j,i} - \theta_0 \Vert \leq (j+1)r_n \\
\therefore \Vert \theta - \theta^{S,j,i}\Vert \leq  \frac{1}{2}\Vert \theta^{S,j,i} - \theta_0 \Vert
$$

따라서 **4)**에서 얻은 upper bound를 대입하면 다음과 같다.

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}} 2\sqrt{\beta_{S,j,i}}  \text{ } e^{-\Vert \theta_0 - \theta^{S,j,i}\Vert^2/32 }\\
\end{align*}
$$

$\theta^{S,j,i} \in \Theta_{S,j,n}$이므로, $2jr_n \leq \Vert \theta_0 - \theta^{S,j,i}\Vert \leq 2(j+1)r_n$이 만족하는 것을 이용하면 다음과 같다.

$$
e^{-\Vert \theta_0 - \theta^{S,j,i}\Vert^2/32 } \leq e^{- 4j^2r_n^2/32 }
$$

따라서 **5)**에서 얻은 $\beta_{S,j,i}$의 upper bound를 대입하면 다음과 같다.

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}} 2\sqrt{\beta_{S,j,i}}  \text{ } e^{- j^2r_n^2/8 }\\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}} 2 \exp\left( \frac{1}{2}\vert S\vert \log j + \frac{1}{2}C (\vert S\vert +\vert S_0\vert)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 } \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M} 2 \exp\left( \log N_{S.j} + \frac{1}{2}\vert S\vert \log j + \frac{1}{2}C (\vert S\vert +\vert S_0\vert)\log n +\frac{1}{2}C^\prime r_n^2 \right)  e^{- j^2r_n^2/8 }\\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M} 2 \exp\left( Aq_n \log C + \frac{1}{2}Aq_n \log j + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 } \\
&= \sum_{j \geq M} 2 \vert \mathcal S_n \vert \exp\left( Aq_n \log C + \frac{1}{2}Aq_n \log j + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 }\\
&\leq \sum_{j \geq M} 2 Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}Aq_n \log j + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 }
\end{align*}
$$

네 번째 부등호는 $\vert S \vert \leq Aq_n , \vert S_0 \vert = q_n$, 그리고 $N_{S,j} \leq C^{\vert S \vert} \leq C^{Aq_n}$을 이용한 것이다. 마지막 부등호는 다음 부등식을 이용한 것이다.

$$
\vert \mathcal S_n \vert \leq Aq_n {n\choose Aq_n}\leq Aq_n e^{ Aq_n \log (ne/Aq_n)}
$$

마지막 줄의 우변을 $j$에 대해 정리하면 다음과 같다.

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq  2 Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right)\sum_{j \geq M} e^{(Aq_n/2) \log j - j^2r_n^2/8 } \\
&=  2 Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right)\sum_{j \geq M} j^{Aq_n/2}e^{- j^2r_n^2/8 } \\
&=  2 Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime q_n \log n \right) \\
&\quad \enspace \times \sum_{j \geq M} j^{Aq_n/2}e^{- j^2q_n \log n/8 }
\end{align*}
$$

마지막 줄의 power series 부분은 다음과 같이 upper bound를 잡을 수 있다.

$$
\begin{align*}
\sum_{j \geq M} j^{Aq_n/2}e^{- j^2q_n \log n/8 } &\leq \sum_{j \geq M} e^{- jq_n \log n/8} \\
&\leq \int_{M-1}^\infty e^{- tq_n \log n/8} dt \\
&\leq \frac{8}{q_n \log n}e^{- (M-1)q_n \log n/8}
\end{align*}
$$

첫 번째 부등호는 모든 $j \geq M$에 대해 $j^2 - j \geq \frac{4A}{\log n}\log j$이 성립하도록 $M$을 설정하면 된다. 이를 적용하면 다음과 같다.

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&=  2 Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime q_n \log n \right) \\
&\quad \enspace \times \sum_{j \geq M} j^{Aq_n/2}e^{- j^2q_n \log n/8 } \\
&\leq  \frac{16 Aq_n}{q_n \log n}\\
&\quad \enspace \times\exp\left( - \frac{M-1}{8}q_n \log n + Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime q_n \log n \right) \\
&\leq  \frac{16 A}{\log n}\\
&\quad \enspace \times\exp\left( \left\{ - \frac{M-1}{8} + A + \frac{1}{2}(C(A+1) + C^\prime)\right\}q_n \log n \right)\\
&\quad \enspace \times \exp\left(A(1-\log A)q_n -Aq_n\log q_n + A\log C q_n\right) \\
\end{align*}
$$

$n$이 클 때, exponential 앞에 곱해진 항은 exponential 항에 dominate되며, exponential의 지수 내에서 $n$에 대해 위 식을 dominate하는 term은 $q_n \log n$이므로, 다음을 만족하도록 $M$을 충분히 크게 설정하면, $n \rightarrow \infty$일 때 마지막 줄의 우변이 0으로 수렴하게 된다.

$$
\text{choose }M\text{ so that } -\frac{M-1}{8} + A + \frac{1}{2}(C(A+1) + C^\prime) < 0
$$

따라서 충분히 큰 $M >0$에 대해 다음이 만족한다.

$$
E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \rightarrow 0, \quad \text{ as }n \rightarrow \infty.\tag{8}
$$

따라서 **3)**의 결과와 식 (6), (7), (8)을 종합하면, 다음과 같이 증명이 끝난다.

$$
\begin{align*}
&E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n \vert y^{(n)} ) \\
&= E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} ) \\
&\enspace+ E_{\theta_0}\Pi_n( \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} ) \\
&\leq  E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} + e^{-{r_n^2}/2} \\
&\enspace+ E_{\theta_0}\Pi_n( \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} ) \\
&\longrightarrow 0, \quad \text{ as }n \rightarrow \infty.
\end{align*}
$$

지금까지는 $a_n = 1/n$을 가정한 상황이었다. $a_n = n^{-(1+\beta)}$의 경우에는 **5)**의 결과를 유도할 때 $q_n \gtrsim \log n$의 가정이 필요하지 않다는 점만 다를 뿐, **5)**의 결과는 동일하다. 따라서 그 이외의 모든 증명 과정은 $a_n = n^{-(1+\beta)}$의 경우에도 동일하게 적용할 수 있다.$\enspace \square$













