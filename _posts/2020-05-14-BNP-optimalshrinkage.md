---
layout: post
title: "Dirichlet–Laplace priors for optimal shrinkage"
tags: [Bayesian Nonparametrics]
comments: true
---



# 0. Setting

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

# 1. Theorem

이 정리는 *Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015). Dirichlet–Laplace priors for optimal shrinkage. Journal of the American Statistical Association.*의 Theorem 3.1이며, 위와 같이 정의한 Dirichlet-Laplace shrinkage prior $\text{DL}_{a_n}$가 **optimal posterior contraction rate**을 갖는다는 것을 보인 theorem이다.



## 1.1. Statement

*Consider model (1) with $\theta \sim \text{DL}_{a_n}$ as in (2), where $a_n = n^{-(1+\beta)}$ for some $\beta >0$ small. Assume $\theta_0 \in l_0[q_n;n]$ with $q_n = o(n)$ and $\Vert \theta_0 \Vert_2^2 \leq q_n \log^4 n$. Then, with $s_n^2 = q_n \log(n/q_n)$ and for some constant $M>0$,*

$$
\lim_{n \rightarrow \infty} E_{\theta_0} \Pi_n( \Vert \theta - \theta_0 \Vert_2 < M s_n \vert y^{(n)} ) = 1 \tag{3}
$$

*If $a_n = 1/n$ instead, then (3) holds when $q_n \gtrsim \log n$.*



## 1.2. Proof

### 1) $P_{\theta_0}(\mathcal A_n^c) \leq e^{-r_n^2/2}$

이 부분은 Castillo and van der Vaart (2012)의 Lemma 5.2에 대한 증명과정이다.

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
\mathcal D_n \geq \Vert \tilde \Pi \Vert \exp\Big(\tilde \mu^T (y-\theta_0) - \frac{1}{2}\tilde \sigma^2 \Big) \tag{4}
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

세 번째 줄의 부등호는 (3)의 결과에 의한 것이다. 

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



### 2) Construction of covering sets of parameter space

$\mathcal S_n$은 다음과 같이 정의된 $\{ 1, 2,\ldots, n\}$의 부분집합의 모임이다. 이 때 $S \in \mathcal S_n$는 $\theta $가 nonzero 값을 갖는 index를 의미한다.

$$
\mathcal S_n = \Big\{ S \subset \{1, 2, \ldots , n\} : \vert S \vert \leq Aq_n \Big\}
$$

한 $S \in \mathcal S_n$와 양의 정수 $j$에 대해, 다음과 같이 $\Theta_{S,j,n}$을 정의하자.

$$
\Theta_{S,j,n} = \Big\{ \theta \in \mathbb R^n : \text{supp}_{\delta_n}(\theta) = S, 2jr_n \leq \Vert \theta-\theta_0 \Vert_2 \leq 2(j+1)r_n\Big\}
$$

이 때 $\delta_n = r_n /n$으로 둔다. 그리고 $\Theta_{S,j,n}$의 $2jr_n$ net을 $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$와 같이 나타내자. 이 때 다음이 만족한다.

$$
\forall \theta \in \Theta_{S,j,n}, \quad \exist i \enspace \text{ s.t. } \Vert \theta - \theta^{S,j,i}\Vert < 2jr_n
$$

그럼 이와 같은 $\Theta_{S,j,n}$의 $jr_n$ net $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$는 어떻게 만들 수 있을까?  

$\vert S\vert$-dimensional ball $\{\Vert \phi - \theta_{0S}\Vert  \leq 2(j+1)r_n\}$의 $\frac{1}{2}jr_n$ net을 $\{ \phi^{S,j,i} : i = 1, \ldots, N_{S,j}\}$라고 하자. Covering number의 volume argument에 의해, 어떤 상수 $C$에 대해 $N_{S,j}$가 다음을 만족하도록 위의 $\vert S\vert$-dimensional ball에 대한 $jr_n$ net을 잡을 수 있다. ([Link](https://www.stat.berkeley.edu/~bartlett/courses/2013spring-stat210b/notes/12notes.pdf) 참고)

$$
N_{S,j} \leq C^{\vert S\vert}
$$

$\theta^{S,j,i}_S = \phi^{S,j,i}$, $\theta_k^{S,j,i} = 0$ for $k \in S^c$와 같이 $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$을 정의하고, 이것이 $\Theta_{S,j,n}$의 $jr_n$ net이 됨을 보이자. 임의의 $\theta \in \Theta_{S,j,n}$를 고르고, 그에 대해 $\Vert \theta_S^{S,j,i} - \theta_S \Vert \leq \frac{1}{2}jr_n$을 만족하는 $1 \leq i\leq N_{S,j}$를 고르자.

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

세 번째 줄의 부등호는 $\Theta_{S,j,n}$의 $S$에의 projection이 중심 $\theta_{0S}$, 반지름 $2(j+1)r_n$의 $\vert S\vert$-dimensional ball의 부분집합이라는 사실에 따른 것이다. 또한 다섯 번째 줄의 부등호는 $\vert S\vert > 1/3$이면 모든 $n$에 대해 만족하는데, $A$를 적절하게 크게 잡으면 이는 문제가 없다.

따라서 위와 같이 construct한 $\{ \theta^{S,j,i} : i = 1, \ldots, N_{S,j}\}$은 $\Theta_{S,j,n}$의 $2jr_n$ net이 된다. 따라서 각 $\theta^{S,j,i}$를 중심으로 갖는 반지름 $2jr_n$의 ball, $B_{S,j,i}$의 union은 $\Theta_{S,j,n}$을 덮는다(cover). 

$$
\Theta_{S,j,n} \subset \bigcup_{i = 1}^{N_{S,j}} B_{S,j,i}
$$


### 3) $E_{\theta_0} \Pi_n(\vert \text{supp}_{\delta_n}(\theta )\vert > Aq_n \vert y^{(n)}) \rightarrow 0$

**(보충 要)**



### 4) Construction of test

이 부분은 Castillo and van der Vaart (2012)의 Lemma 5.1이다. Statement는 다음과 같다.

*For any $\alpha, \beta >0$, and any $\theta_0, \theta_1 \in \mathbb R^n$, there exists a test $\phi$ based on $y \sim \text{N}(\theta, I)$, such that for every $\theta \in \mathbb R^n$ with $\Vert \theta - \theta_1 \Vert \leq \Vert \theta_0 - \theta_1 \Vert/2 \stackrel{\triangle}{=} \rho$,*

$$
\begin{align*}
\alpha E_{\theta_0}\phi + \beta E_\theta(1-\phi) &\leq \alpha \left[ 1- \Phi\Big(\frac{\rho}{2} + \frac{1}{\rho} \log\frac{\alpha}{\beta} \Big)\right] + \beta \Phi\Big(-\frac{\rho}{2} + \frac{1}{\rho} \log\frac{\alpha}{\beta} \Big) \\
&\leq 2\sqrt{\alpha \beta}e^{-\Vert \theta_0 - \theta_1 \Vert^2 /32}
\end{align*}
$$

먼저 $\theta_0 = \mathbf 0$을 가정하고 위 사실을 증명하자.

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

(ii) $ 0 < D-\rho < D$의 경우, $\alpha [ 1-\Phi\left( D\right)] $는 (i)와 동일하게 bound를 잡을 수 있다. $D-\rho > 0 $이므로,

$$
\begin{align*}
D-&\rho = \frac{1}{\rho}\log \frac{\alpha}{\beta} - \frac{1}{2}\rho > 0 \\
\implies \beta &< \alpha e^{-\rho^2/2}
\end{align*}
$$

이를 이용하면,

$$
\begin{align*}
\beta \Phi( D - \rho)&< \alpha e^{-\rho^2/2}\Phi( D - \rho)\\
&asdfasdfasdfasdfasdf
\end{align*}
$$

**(보충 要)**

(iii) $D- \rho \leq D < 0$의 경우, $\beta\Phi( D - \rho)$는 (i)와 동일하게 bound를 잡을 수 있다. $D < 0 $이므로,

$$
\begin{align*}
D& = \frac{1}{\rho}\log \frac{\alpha}{\beta} + \frac{1}{2}\rho < 0 \\
\implies \alpha &< \beta e^{-\rho^2/2}
\end{align*}
$$

이를 이용하면,

$$
\begin{align*}
\alpha \Big[ 1- \Phi( D )\Big]&< \beta e^{-\rho^2/2}\Big[ 1- \Phi( D )\Big]\\
&asdfasdfasdfasdfasdf
\end{align*}
$$

**(보충 要)**

따라서 (i), (ii), (iii) 모든 경우에서 위 bound가 성립하므로, $\Vert \theta_1 \Vert/2 = \rho$를 이용하면 다음과 같다.

$$
\begin{align*}
\therefore \alpha E_{\theta_0}\phi + \beta E_\theta (1-\phi) &\leq 2 \sqrt{\alpha\beta} \text{ }e^{- \left( \log \frac{\alpha}{\beta}\right)^2/2\rho^2}e^{-{\rho^2}/{8}} \\ 
&\leq 2 \sqrt{\alpha\beta} \text{ }e^{-{\rho^2}/{8}} \\
&= 2 \sqrt{\alpha\beta} \text{ }e^{-{\Vert \theta_1 \Vert^2}/{36}} \\ 
\end{align*}
$$


**왜 $\theta_0 = \mathbf 0$로 두고 해도 무방?  ** **(보충 要)**

 

### 5) Upper bound of $\beta_{S,j,i}$

**(보충 要)**



### 6) The main result

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
&\enspace+ E_{\theta_0}\Pi_n( \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} ) \tag{5}
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
&\leq  E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} + e^{-{r_n^2}/2} \tag{6}
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
&\leq E_{\theta_0}\left[ \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}\Pi_n( B^{S,j,i}\vert y^{(n)} )I_{\mathcal A_n}\right] \\
\end{align*}
$$

두 번째 등호는 위에서 정의한 $\Theta_{S,j,n}$을 사용한 것이며, 마지막 줄의 부등호는 위에서와 같이 construct한 $\Theta_{S,j,n}$의 covering $\{B^{S,j,i}: i=1, \ldots , N_{S,j}\}$를 사용한 것이다. 

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq E_{\theta_0}\left[ \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}\Pi_n( B^{S,j,i}\vert y^{(n)} )I_{\mathcal A_n}\right] \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}} E_{\theta_0} \left[ \left\{ \phi_{S,j,i}\Pi_n( B^{S,j,i}\vert y^{(n)} ) + (1-\phi_{S,j,i})\Pi_n( B^{S,j,i}\vert y^{(n)} )\right\}I_{\mathcal A_n}\right] \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})\Pi_n( B^{S,j,i}\vert y^{(n)} )I_{\mathcal A_n}\right]\right\} \\

\end{align*}
$$

Posterior probability의 정의를 사용하면 $\Pi_n( B^{S,j,i}\vert y^{(n)} )$은 다음과 같다.

$$
\Pi_n( B^{S,j,i}\vert y^{(n)} ) = \frac{\int_{B^{S,j,i}} \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}{\int \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}
$$

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\

&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B^{S,j,i}} \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}{\int \prod_{i=1}^{n}f_{\theta_i}(y_i) d\Pi(\theta)}
I_{\mathcal A_n}\right]\right\} \\

&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B^{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{\int \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}
I_{\mathcal A_n}\right]\right\} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B^{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{\mathcal D_n}
I_{\mathcal A_n}\right]\right\} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B^{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\right]\right\} \\
\end{align*}
$$

마지막 부등호는 사건 $\mathcal A_n$ 위에서 $\mathcal D_n \geq e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)$가 성립함을 이용한 것이다. 또한 다음과 같은 사실을 이용할 수 있다.

$$
\begin{align*}
&E_{\theta_0}\left[ (1-\phi_{S,j,i})
\frac{\int_{B^{S,j,i}} \prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)} d\Pi(\theta)}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\right] \\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\iint_{B^{S,j,i}}(1-\phi_{S,j,i}) \left[\prod_{i=1}^{n}\frac{f_{\theta_i}(y_i)}{f_{\theta_{0i}}(y_i)}\right]\left[\prod_{i=1}^{n}{f_{\theta_{0i}}(y_i)}\right]d\Pi(\theta)dy\\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\iint_{B^{S,j,i}}(1-\phi_{S,j,i}) \left[\prod_{i=1}^{n}{f_{\theta_i}(y_i)}\right]d\Pi(\theta)dy\\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\int_{B^{S,j,i}}\int(1-\phi_{S,j,i}) \left[\prod_{i=1}^{n}{f_{\theta_i}(y_i)}\right]dyd\Pi(\theta)\\
&= \frac{1}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\int_{B^{S,j,i}}E_\theta (1-\phi_{S,j,i}) d\Pi(\theta)\\
&\leq \frac{\Pi(B^{S,j,i})}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\sup_{\theta \in B^{S,j,i}} \Big\{ E_\theta (1-\phi_{S,j,i})\Big\}\\
\end{align*}
$$

이를 대입하면,

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ \frac{\Pi(B^{S,j,i})}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}
\sup_{\theta \in B^{S,j,i}} \Big\{ E_\theta (1-\phi_{S,j,i})\Big\}\right\} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}}  \left\{ E_{\theta_0} \phi_{S,j,i}+ \beta_{S,j,i}
\sup_{\theta \in B^{S,j,i}} \Big\{ E_\theta (1-\phi_{S,j,i})\Big\}\right\} \\
\end{align*}
$$

이 때 $ \beta_{S,j,i}= \frac{\Pi(B^{S,j,i})}{e^{-4r_n^2} \Pi(\Vert \theta - \theta_0\Vert_2 \leq 2r_n)}$이다.  

$\theta_0 = \theta_0, \theta_1 = \theta^{S,j,i}$로 두면, 다음과 같이 **4)**의 조건이 만족하는 것을 확인할 수 있다.

$$
\theta \in B^{S,j,i} \implies \Vert \theta - \theta^{S,j,i}\Vert \leq jr_n \\
 \theta^{S,j,i} \in \Theta_{S,j,n} \implies  jr_n \leq \frac{1}{2}\Vert \theta^{S,j,i} - \theta_0 \Vert \leq (j+1)r_n \\
\therefore \Vert \theta - \theta^{S,j,i}\Vert \leq  \frac{1}{2}\Vert \theta^{S,j,i} - \theta_0 \Vert
$$

따라서 **4)**에서 얻은 upper bound를 대입하면 다음과 같다.

$$
\begin{align*}
& E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}} 2\sqrt{\beta_{S,j,i}}  \text{ } e^{-\Vert \theta_0 - \theta^{S,j,i}\Vert^2/32 }\\
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
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M}\sum_{i = 1}^{N_{S,j}} 2 \exp\left( \frac{1}{2}\vert S\vert \log2j + \frac{1}{2}C (\vert S\vert +\vert S_0\vert)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 } \\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M} 2 \exp\left( \log N_{S.j} + \frac{1}{2}\vert S\vert \log2j + \frac{1}{2}C (\vert S\vert +\vert S_0\vert)\log n +\frac{1}{2}C^\prime r_n^2 \right)  e^{- j^2r_n^2/8 }\\
&\leq \sum_{S \in \mathcal S_n}\sum_{j \geq M} 2 \exp\left( Aq_n \log C + \frac{1}{2}Aq_n \log2j + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 } \\
&= \sum_{j \geq M} 2 \vert \mathcal S_n \vert \exp\left( Aq_n \log C + \frac{1}{2}Aq_n \log2j + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 }\\
&\leq \sum_{j \geq M} 2 Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}Aq_n \log2j + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right) e^{- j^2r_n^2/8 }
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
&\leq  2 Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right)\sum_{j \geq M} e^{(Aq_n/2) \log2j - j^2r_n^2/8 } \\
&=  2^{Aq_n/2 + 1} Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime r_n^2 \right)\sum_{j \geq M} j^{Aq_n/2}e^{- j^2r_n^2/8 } \\
&=  2^{Aq_n/2 + 1} Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime q_n \log n \right) \\
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
&=  2^{Aq_n/2 + 1} Aq_n \exp\left( Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime q_n \log n \right) \\
&\quad \enspace \times \sum_{j \geq M} j^{Aq_n/2}e^{- j^2q_n \log n/8 } \\
&\leq  \frac{2^{Aq_n/2 + 4} Aq_n}{q_n \log n}\\
&\quad \enspace \times\exp\left( - \frac{M-1}{8}q_n \log n + Aq_n \log (ne/Aq_n)+ Aq_n \log C + \frac{1}{2}C (Aq_n +q_n)\log n +\frac{1}{2}C^\prime q_n \log n \right) \\
&\leq  \frac{2^{Aq_n/2 + 4} A}{\log n}\\
&\quad \enspace \times\exp\left( \left\{ - \frac{M-1}{8} + A + \frac{1}{2}(C(A+1) + C^\prime)\right\}q_n \log n \right)\\
&\quad \enspace \times \exp\left(A(1-\log A)q_n -Aq_n\log q_n + A\log C q_n\right) \\
\end{align*}
$$

$n$이 클 때 exponential 앞에 곱해진 항은 exponential 항에 dominate되며, exponential의 지수 내에서 $n$에 대해 위 식을 dominate하는 term은 $q_n \log n$이므로, 다음을 만족하도록 $M$을 충분히 크게 설정하면, $n \rightarrow \infty$일 때 마지막 줄의 우변이 0으로 수렴하게 된다.

$$
\text{choose }M\text{ so that } -\frac{M-1}{8} + A + \frac{1}{2}(C(A+1) + C^\prime) < 0
$$

따라서 충분히 큰 $M >0$에 대해 다음이 만족한다.

$$
E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} \rightarrow 0, \quad \text{ as }n \rightarrow \infty.\tag{7}
$$

따라서 **3)**의 결과와 식 (5), (6), (7)을 종합하면, 다음과 같이 증명이 끝난다.

$$
\begin{align*}
&E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n \vert y^{(n)} ) \\
&= E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > M s_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} ) \\
&\enspace+ E_{\theta_0}\Pi_n( \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} ) \\
&\leq  E_{\theta_0}\Pi_n( \Vert \theta - \theta_0 \Vert_2 > 2M r_n , \vert \text{supp}_{\delta_n}(\theta)\vert \leq Aq_n \vert y^{(n)} )I_{\mathcal A_n} + e^{-{r_n^2}/2} \\
&\enspace+ E_{\theta_0}\Pi_n( \vert \text{supp}_{\delta_n}(\theta)\vert > Aq_n \vert y^{(n)} ) \\
\\
&\longrightarrow 0, \quad \text{ as }n \rightarrow \infty.
\end{align*}
$$

지금까지는 $a_n = 1/n$을 가정한 상황이었다. $a_n = n^{-(1+\beta)}$의 경우에는 **5)**의 결과를 유도할 때 $q_n \gtrsim \log n$의 가정이 필요하지 않다는 점만 다를 뿐, **5)**의 결과는 동일하다. 따라서 그 이외의 모든 증명 과정은 $a_n = n^{-(1+\beta)}$의 경우에도 동일하게 적용할 수 있다.$\enspace \square$













