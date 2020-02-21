---
layout: post
title: "Sampling Methods for Dirichlet Process Mixture Models under Non-conjugate Priors"
tags: [Bayesian Nonparametrics]
comments: true
---

## 0. Setting

$$
\begin{align*}
y_i \vert \mathbf c, \phi &\stackrel{\text{ind}}{\sim} f(y_i \vert \phi_{c_i}) \\
\phi_c &\stackrel{\text{ind}}{\sim} G_0 \\
\mathbf c &\sim p(\mathbf c) \\
\text{where }p(\mathbf c) &= \frac{\alpha^K \prod_{j=1}^{K}(n_j-1)!}{(\alpha)_{n\uparrow}K!}.
\end{align*}
$$

이를 $\theta_i = \phi_{c_i}$와 Dirichlet process prior에서 생성된 $P$를 이용하여 나타내면 다음과 같다.

$$
\begin{align*}
y_i \vert \theta_i &\stackrel{\text{ind}}{\sim} f(y_i \vert \theta_i) \\
\theta_i \vert P &\stackrel{\text{iid}}{\sim} P \\
P &\sim \text{DP}(\alpha, G_0)
\end{align*}
$$




## 1. Algorithm 4 from Neal (2000) : "no gaps" algorithm

"No gaps" 가정은 $K$개의 서로 다른 종류의 label을 갖는 $\mathbf c =(c_1, \cdots, c_n)$이 $1\sim K$의 label을 모두, gap 없이, 갖는다는 가정이다. 이를 이용하여 모수 $\phi$의 prior $G_0$이 non-conjugate인 상황에서 Dirichlet process mixture model의 posterior를 추정하는 알고리즘을 도출할 수 있다.

### Algorithm

Markov chain의 state는 $\mathbf c = (c_1, \cdots, c_n), \phi = \{ \phi_c : c \in \mathbf{c} \}$이다. 이를 다음과 같이 반복하여 sampling한다.

* For $i=1,\cdots, n:$

  *  $c_i$를 제외한 $\mathbf{c}_{-i}$의 서로 다른 label의 개수를 $k^-$라고 하자. $c_j \in \mathbf{c}_{-i}$는 $\{1,\cdots, k^-\}$의 값을 갖는다. 
  * 만약 모든 $j \neq i$에 대해 $c_i \neq c_j$ 가 성립한다면, 즉 $c_i$의 label이 $\mathbf{c}_{-i}$에는 없는 "singleton" label이라면, $1/(k^- +1)$의 확률로 $c_i$를 $k^- +1$로 relabel한다. 그에 따라 $\phi_{c_i}$ 역시 $\phi_{k^- +1}$로 재지정된다.
  * 만약 어떤 $j \neq i$에 대해 $c_i = c_j$가 성립한다면, 즉 $c_i$와 같은 label이 $\mathbf{c}_{-i}$에 있다면, $G_0$으로부터 새로 $\phi_{k^- +1}$을 sampling한다.
  * 다음 확률에 따라 $c_i$의 새 value를 생성한다.

  $$
  p(c_i = c \vert \mathbf{c}_{-i}, y_i, \phi) \propto 
  \begin{cases}
  n_{-i,c} f(y_i \vert \phi_c) \quad  \text{if }1 \leq c \leq k^- \\
  \\
  \frac{\alpha}{k^- +1} f(y_i \vert \phi_c) \quad  \text{if }c = k^- +1\\
  \end{cases}
  $$

  * Observation $y_i$와 연결된 $\phi_c$들만을 state에 남기고, 선택되지 않은 label들은 삭제한다.

* For all $c \in \mathbf{c}:$

  * 다음 분포로부터 $\phi_c$의 새 value를 생성한다.

  $$
  p(\text{d}\phi_c \vert \phi_{-c}, \mathbf{c}, y) \propto  \left[ \prod_{i: c_i = c} f(y_i \vert \phi_{c_i}) \right] G_0(\text{d} \phi_c)
  $$



### Derivation

$\mathbf c, \phi$에 대한 posterior distribution은 다음과 같다.
$$
\begin{align*}
p(\text{d} \mathbf c, \text{d}\phi \vert y) &\propto p(\text{d} \mathbf c, \text{d}\phi, \text{d} y) \\
& = p(\text{d} y \vert \mathbf c, \phi)p(\text{d} \mathbf c)p(\text{d}\phi) \\
& = \left[ \prod_{i=1}^{n} f(y_i \vert \phi_{c_i}) \right] p(\text{d} \mathbf c)\left[ \prod_{j=1}^{K} G_0(\text{d} \phi_j) \right]
\end{align*}
$$

### Updating $\mathbf c$

$c_i$를 update하기 위해 conditional을 구해보면 다음과 같다.

$$
p(c_i = c \vert \mathbf{c}_{-i}, \phi, y) \propto f(y_i \vert \phi_c)p(c_i =c \vert \mathbf{c}_{-i})
$$

먼저 모든 $j \neq i$에 대해 $c_i \neq c_j$ 가 성립하는 "singleton"의 경우를 생각해보자. $\mathbf c_{-i}$의 gap을 메우는 $c_i$의 label 값이 $c_0$라고 하자. 그 때 다음이 성립한다.

$$
p(c_i = c_0 \vert \mathbf{c}_{-i}) = \frac{p(c_i = c_0 , \mathbf{c}_{-i})}{p( \mathbf{c}_{-i})} \stackrel{\text{no gap}}{=} \frac{p(c_i = c_0 , \mathbf{c}_{-i})}{p( c_i = c_0, \mathbf{c}_{-i})} = 1 \\
p(c_i = c \vert \mathbf{c}_{-i}) = 0, \quad \forall c \neq c_0
$$

이 때 우리는  $1/(k^- +1)$의 확률로 $c_i$를 $k^- +1$로 relabel한다.  따라서 다음과 같이 알고리즘이 도출된다.

$$
\begin{align*}
p(c_i = c \vert \mathbf{c}_{-i}, y_i, \phi) &\propto f(y_i \vert \phi_{c})p(c_i =c \vert \mathbf{c}_{-i})\\
&=
\begin{cases}
n_{-i,c} f(y_i \vert \phi_{c}) \quad \quad  \text{if }1 \leq c \leq k^- \text{ and }c \neq c_0\\
\frac{k^-}{k^- +1} n_{-i,c_0}f(y_i \vert \phi_{c_0}) \quad  \text{if }c = c_0\\
\frac{\alpha}{k^- +1} f(y_i \vert \phi_{c}) \quad \quad  \text{if }c = k^- +1\\
\end{cases} \\
\\
&=
\begin{cases}
n_{-i,c} f(y_i \vert \phi_{c}) \quad  \quad  \text{if }1 \leq c \leq k^- \text{ and }c \neq c_0\\
n_{-i,c_0}f(y_i \vert \phi_{c_0}) \quad \quad  \text{if }c = c_0 \quad (\because n_{-i,c_0}=0)\\
\frac{1}{k^- +1} \alpha f(y_i \vert \phi_{c}) \quad \quad  \text{if }c = k^- +1\\
\end{cases} \\
\\
&=
\begin{cases}
n_{-i,c} f(y_i \vert \phi_{c}) \quad  \quad  \text{if }1 \leq c \leq k^- \\
\\
\frac{\alpha}{k^- +1} f(y_i \vert \phi_{c}) \quad \quad  \text{if }c = k^- +1\\
\end{cases} \\
\end{align*}
$$

어떤 $j \neq i$에 대해 $c_i = c_j$가 성립하는 경우에 대해서도 생각해보자. $1 \leq c \leq k^-$에 대해서는 $c_i$의 conditional prior가 다음과 같다.

$$
\begin{align*}
p(c_i = c \vert \mathbf{c}_{-i}) &\propto p(c_i =c , \mathbf{c}_{-i})\\
&=\frac{\alpha^{k^-}n_{-i,c}! \prod_{\ell \neq c}(n_{-i,\ell}-1)!}{(\alpha)_{n\uparrow}k^- !} \quad \quad  \text{if }1 \leq c \leq k^-\\
\end{align*}
$$

$c = k^- +1$에 대해서는 다음과 같다.

$$
\begin{align*}
p(c_i = c \vert \mathbf{c}_{-i}) &\propto p(c_i =c , \mathbf{c}_{-i})\\
&=\frac{\alpha^{k^- +1} \prod_{\ell =1}^{k^-}(n_{-i,\ell}-1)!}{(\alpha)_{n\uparrow}(k^-+1) !} \quad \quad  \text{if }c = k^- +1
\\
p(c_i = c \vert \mathbf{c}_{-i}) &\propto
\begin{cases}
n_{-i,c}  \quad  \quad  \text{if }1 \leq c \leq k^- \\
\\
\frac{\alpha}{k^- +1}  \quad \quad  \text{if }c = k^- +1\\
\end{cases} \\
\end{align*}
$$

따라서, 이를 정리하면 다음과 같이 $c_i$를 update하는 알고리즘이 도출된다.

$$
\begin{align*}
p(c_i = c \vert \mathbf{c}_{-i}) &\propto\begin{cases}n_{-i,c}  \quad  \quad  \text{if }1 \leq c \leq k^- \\\\\frac{\alpha}{k^- +1}  \quad \quad  \text{if }c = k^- +1\\
\end{cases} \\
\\
p(c_i = c \vert \mathbf{c}_{-i}, y_i, \phi) &\propto f(y_i \vert \phi_{c})p(c_i =c \vert \mathbf{c}_{-i})\\
&=
\begin{cases}
n_{-i,c} f(y_i \vert \phi_{c}) \quad  \quad  \text{if }1 \leq c \leq k^- \\
\\
\frac{\alpha}{k^- +1} f(y_i \vert \phi_{c}) \quad \quad  \text{if }c = k^- +1\\
\end{cases}
\end{align*}
$$

### Updating $\phi$

$\phi_c$의 conditional은 다음과 같이 구할 수 있다.

$$
\begin{align*}
p(\text{d}\phi_c \vert \phi_{-c}, \mathbf{c}, y) &\propto p(\text{d} \mathbf c, \text{d}\phi, \text{d} y) \\
& = p(\text{d} y \vert \mathbf c, \phi)p(\text{d} \mathbf c)p(\text{d}\phi) \\
& = \left[ \prod_{i=1}^{n} f(y_i \vert \phi_{c_i}) \right] p(\text{d} \mathbf c)\left[ \prod_{j=1}^{K} G_0(\text{d} \phi_j) \right] \\
&\propto \left[ \prod_{i: c_i = c} f(y_i \vert \phi_{c_i}) \right] G_0(\text{d} \phi_c)
\end{align*}
$$


## 2. Algorithm 5 from Neal (2000)

이 알고리즘은 Metropolis-Hastings update를 이용하여 $\mathbf c$의 각 component, $c_i$를 update한다. Target distribution은 $c_i$의 conditional $p(c_i = c \vert \mathbf c_{-i}, y_i, \phi)$이며, proposal distribution $Q(c^\ast \vert c)$는 $c_i$의 conditional prior를 사용한다.

$$
Q(c^\ast \vert c) =p(c_i = c^\ast \vert \mathbf{c}_{-i})
$$

이 때 Metropolis-Hastings update의 acceptance probability $a(c^\ast, c)$는 다음과 같다.

$$
\begin{align*}
a(c^\ast, c) &= \min \left[ 1, \frac{p(c_i = c^\ast \vert \mathbf{c}_{-i}, y_i, \phi)}{p(c_i = c \vert \mathbf{c}_{-i}, y_i, \phi)} \frac{Q(c \vert c^\ast)}{Q(c^\ast \vert c)} \right] \\
&= \min \left[ 1, \frac{f(y_i \vert \phi_{c^\ast})p(c_i =c^\ast \vert \mathbf{c}_{-i})}{f(y_i \vert \phi_{c})p(c_i =c \vert \mathbf{c}_{-i})} \frac{p(c_i = c \vert \mathbf{c}_{-i})}{p(c_i = c^\ast \vert \mathbf{c}_{-i})} \right] \\
&= \min \left[ 1, \frac{f(y_i \vert \phi_{c^\ast})}{f(y_i \vert \phi_{c})}\right] \\ 
\end{align*}
$$

### Algorithm

Markov chain의 state는 $\mathbf c = (c_1, \cdots, c_n), \phi = \{ \phi_c : c \in \mathbf{c} \}$이다. 이를 다음과 같이 반복하여 sampling한다.

* For $i=1,\cdots, n:$  모든 $i$에 대해 아래와 같은 $c_i$의 update를 $R$번씩 반복한다.

  * Proposal distribution $Q(c^\ast \vert c) =p(c_i = c^\ast \vert \mathbf{c}_{-i})$으로부터 candidate $c^\ast$를 생성한다.
    * 만약 $c^\ast$가 기존의 $\mathbf c$에 속하지 않는 새로운 label이라면, $G_0$으로부터 그에 대한 새 $\phi_{c^\ast}$를 생성해준다.
  * Acceptance probability $a(c^\ast, c)$에 따라 이를 accept/reject한다.
    * reject되었다면 $c_i$의 새 value는 기존의 value와 같은 값을 갖는다.

* For all $c \in \mathbf{c} = \{c_1, \cdots, c_n \}:$

  * 다음 분포로부터 $\phi_c$의 새 value를 생성한다.

  $$
  p(\text{d}\phi_c \vert \phi_{-c}, \mathbf{c}, y) \propto  \left[ \prod_{i: c_i = c} f(y_i \vert \phi_{c_i}) \right] G_0(\text{d} \phi_c)
  $$



## 3. Algorithm 6 from Neal (2000)

이 알고리즘은 Algorithm 5와 마찬가지로 Metropolis-Hastings update을 이용한 알고리즘이다. Markov chain의 state를 $\mathbf c, \phi$가 아닌 $\theta = (\theta_1, \cdots, \theta_n)$으로 두고 수행한다는 점에서 Algorithm 5와 차이가 있다.

$$
Q(\theta^\ast \vert \theta) =p(\theta_i = \theta^\ast \vert \mathbf{\theta}_{-i}) \propto 
\begin{cases}
n_{-i,\theta^\ast} \quad \quad \text{ if } \theta^\ast \in \{\theta_1, \cdots, \theta_n\}\\
\\
\alpha \quad \quad \text{ otherwise } \\
\end{cases} \\
\theta_i \vert \mathbf{\theta}_{-i} \sim \frac{1}{n-1+\alpha}\sum_{j\neq i}\delta_{\theta_j}(\cdot) + \frac{\alpha}{n-1+\alpha}G_0(\cdot)
$$

이 때 Metropolis-Hastings update의 acceptance probability $a(\theta^\ast, \theta)$는 다음과 같다.

$$
\begin{align*}
a(c^\ast, c) &= \min \left[ 1, \frac{p(\theta_i = \theta^\ast \vert \theta_{-i}, y_i)}{p(\theta_i = \theta \vert \theta_{-i}, y_i)} \frac{Q(\theta \vert \theta^\ast)}{Q(\theta^\ast \vert \theta)} \right] \\
&= \min \left[ 1, \frac{f(y_i \vert \theta^\ast)p(\theta_i =\theta^\ast \vert \theta_{-i})}{f(y_i \vert \theta)p(\theta_i =\theta \vert \theta_{-i})} \frac{p(\theta_i = \theta \vert \theta_{-i})}{p(\theta_i = \theta^\ast \vert \theta_{-i})} \right] \\
&= \min \left[ 1, \frac{f(y_i \vert \theta^\ast)}{f(y_i \vert \theta)}\right] \\
\end{align*}
$$

### Algorithm

Markov chain의 state는 $\theta = (\theta_1, \cdots, \theta_n)$이다. 다음과 같이 반복하여 $\theta$를 sampling한다.

* For $i=1,\cdots, n:$  모든 $i$에 대해 아래와 같은 $\theta_i $의 update를 $R$번씩 반복한다.
  * Proposal distribution $Q(\theta^\ast \vert \theta) =p(\theta_i = \theta^\ast \vert \mathbf{\theta}_{-i})$으로부터 candidate $\theta^\ast$를 생성한다.
  * Acceptance probability $a(\theta^\ast, \theta )$에 따라 이를 accept/reject한다.
    * reject되었다면 $\theta_i $의 새 value는 기존의 value와 같은 값을 갖는다.



## 4. Algorithm 7 from Neal (2000)

이 알고리즘은 위 두 알고리즘에서 사용한 Metropolis-Hastings update가 새 component를 더 자주 탐색하도록 proposal distribution에 약간의 수정을 더한 알고리즘이다. 이 proposal distribution은 $c_i$가 singleton인 경우와 그렇지 않은 경우에 따라 다른 분포를 갖는다.

$$
\text{If }c_i \text{ is not a singleton, }Q(c^\ast \vert c) = 
\begin{cases} 
0 \quad \text{ if }c^\ast \in \mathbf c_{-i}\\
\\
1 \quad \text{ if }c^\ast \notin \mathbf c_{-i}\\
\end{cases},\\

\text{if }c_i \text{ is a singleton, }Q(c^\ast \vert c) = 
\begin{cases} 
\frac{n_{-i,c^\ast}}{n-1} \quad \text{ if }c^\ast \in \mathbf c_{-i}\\
\\
0 \quad \text{ if }c^\ast \notin \mathbf c_{-i}\\
\end{cases}.\\
$$


### Algorithm

Markov chain의 state는 $\mathbf c = (c_1, \cdots, c_n), \phi = \{ \phi_c : c \in \mathbf{c} \}$이다. 이를 다음과 같이 반복하여 sampling한다.

* M-H updates : For $i=1,\cdots, n, $

  * 만약 $c_i$가 singleton이 아니라면, proposal distribution $Q(c^\ast \vert c)$으로부터 candidate $c^\ast$를 생성하며, 이 때의 $c^\ast$는 새 label이다. 아래와 같은 acceptance probability $a (c^\ast, c)$에 따라 accept/reject 여부를 결정하고, accept하게 되었다면 $c^\ast$를 새 label로 지정하고, 그에 대한 parameter $\phi_{c^\ast}$를 $G_0$로부터 생성한다. 

  $$
  \begin{align*}
  a(c^\ast, c) &= \min \left[ 1, \frac{p(c_i = c^\ast \vert \mathbf{c}_{-i}, y_i, \phi)}{p(c_i = c \vert \mathbf{c}_{-i}, y_i, \phi)} \frac{Q(c \vert c^\ast)}{Q(c^\ast \vert c)} \right] \\
  &= \min \left[ 1, \frac{f(y_i \vert \phi_{c^\ast})p(c_i =c^\ast \vert \mathbf{c}_{-i})}{f(y_i \vert \phi_{c})p(c_i =c \vert \mathbf{c}_{-i})} \frac{Q(c \vert c^\ast)}{Q(c^\ast \vert c)} \right] \\
  &= \min \left[ 1, \frac{f(y_i \vert \phi_{c^\ast})}{f(y_i \vert \phi_{c})} \frac{\alpha/(n-1+\alpha)}{n_{-i,c}/(n-1+\alpha)} \frac{n_{-i,c}/(n-1)}{1} \right] \\
  &= \min \left[ 1, \frac{\alpha}{n-1}\frac{f(y_i \vert \phi_{c^\ast})}{f(y_i \vert \phi_{c})}\right] \\ 
  \end{align*}
  $$

  * 만약 $c_i$가 singleton이라면, proposal distribution $Q(c^\ast \vert c)$으로부터 생성된 candidate $c^\ast$는 기존에 존재하던 label이다. 아래와 같은 acceptance probability $a (c^\ast, c)$에 따라 accept/reject 여부를 결정하고, accept하게 되었다면 $c_i, \phi_{c_i}$를 $c^\ast, \phi_{c^\ast}$로 update한다. 

  $$
  \begin{align*}
  a(c^\ast, c) &= \min \left[ 1, \frac{p(c_i = c^\ast \vert \mathbf{c}_{-i}, y_i, \phi)}{p(c_i = c \vert \mathbf{c}_{-i}, y_i, \phi)} \frac{Q(c \vert c^\ast)}{Q(c^\ast \vert c)} \right] \\
  &= \min \left[ 1, \frac{f(y_i \vert \phi_{c^\ast})p(c_i =c^\ast \vert \mathbf{c}_{-i})}{f(y_i \vert \phi_{c})p(c_i =c \vert \mathbf{c}_{-i})} \frac{Q(c \vert c^\ast)}{Q(c^\ast \vert c)} \right] \\
  &= \min \left[ 1, \frac{f(y_i \vert \phi_{c^\ast})}{f(y_i \vert \phi_{c})} \frac{n_{-i,c}/(n-1+\alpha)}{\alpha/(n-1+\alpha)} \frac{1}{n_{-i,c}/(n-1)} \right] \\
  &= \min \left[ 1, \frac{n-1}{\alpha} \frac{f(y_i \vert \phi_{c^\ast})}{f(y_i \vert \phi_{c})}\right] \\ 
  \end{align*}
  $$
  * Reject되었다면 $c_i$의 새 value는 기존의 value와 같은 값을 갖는다.

* Partial Gibbs sampling : For $i=1,\cdots, n, $
  * $c_i$가 singleton이라면, 가만히 둔다.
  * $c_i$가 singleton이 아니라면, 다음 확률에 따라 $c_i$를 relabeling한다.

  $$
  \begin{align*}
  p(c_i = c \vert \mathbf c_{-i}, y_i, \phi, c_i \in \{ c_1, \cdots, c_n \} ) 
  &\propto p(c_i = c \vert \mathbf c_{-i},   c_i \in \{ c_1, \cdots, c_n \} , \phi) p(y_i \vert c_i = c, \phi_c ) \\
  &\propto \frac{n_{-i,c}}{n-1} f(y_i\vert \phi_c) \\
  \end{align*}
  $$

* For all $c \in \mathbf{c} = \{c_1, \cdots, c_n \}:$

  * 다음 분포로부터 $\phi_c$의 새 value를 생성한다.

  $$
  p(\text{d}\phi_c \vert \phi_{-c}, \mathbf{c}, y) \propto  \left[ \prod_{i: c_i = c} f(y_i \vert \phi_{c_i}) \right] G_0(\text{d} \phi_c)
  $$



## 5. Algorithm 8 from Neal (2000)

2~4.에서 소개한 Algorithm 5~7은 Metropolis-Hastings update를 이용하여 sampling을 수행했다. Algorithm 8은 auxiliary variable을 이용해서 sampling을 수행하는 알고리즘이다. Auxiliary variable을 이용한 sampling을 간단히 소개하자면, target distribution $\pi_x$를 marginal 분포로 갖는 joint distribution $\pi_{xy}$에서 sampling을 수행하고, sampling이 끝나면 auxiliary variable인 $y$를 버리고 $x$의 값만 취하는 방법이다.

여기서는 $G_0$에 대해 적분을 수행하지 않고 Dirichlet process mixture model의 sampling을 수행하기 위해, **아직 어느 observation과도 연결되지 않은 label$(c)$의 parameter$(\phi)$의 가능한 값**들을 나타내는 auxiliary variable을 도입한다. 그리고 이 auxiliary variable을 포함한 분포에 대한 gibbs sampling을 통해 $c_i$를 update한다.



### Algorithm

Markov chain의 state는 $\mathbf c = (c_1, \cdots, c_n), \phi = \{ \phi_c : c \in \mathbf{c} \}$이다. 이를 다음과 같이 반복하여 sampling한다. 아래 알고리즘에서, $m$은 새 $\phi$의 candidate으로 둘 auxiliary variable의 수를 나타내는 hyperparameter이다.

* For $i=1,\cdots, n: $

  * $\mathbf c_{-i}$의 서로 다른 label의 수를 나타내는 $k^-$에 대해 $h = k^- + m$를 정의한다.
  * 다음과 같이 $\phi_1, \cdots, \phi_{k^-}, \phi_{k^- +1}, \cdots, \phi_h$를 지정한다.
    * 만약 $c_i$가 singleton이 아니라면, $\phi_{k^- +1}, \cdots, \phi_h$를 $G_0$에서 independent하게 생성한다.
    * 만약 $c_i$가 singleton이라면, $\phi_{k^- +1}$을 $\phi_{c_i}$로 지정해주고$(c_i \stackrel{\text{set}}{=} k^- +1)$, $\phi_{k^- +2}, \cdots, \phi_h$를 $G_0$에서 independent하게 생성한다.
  * $c_i$의 새 value를 다음 확률에 따라 update한다.

  $$
  \begin{align*}
  p(c_i = c \vert \mathbf c_{-i}, y_i, \phi_1, \cdots, \phi_h) &\propto 
  p(c_i = c \vert \mathbf c_{-i}, \phi_1, \cdots, \phi_h) 
  p( y_i \vert c_i = c, \phi_1, \cdots, \phi_h)\\
  
  &\propto 
  \begin{cases}
  \frac{n_{-i,c}}{n-1+\alpha} f(y_i \vert \phi_c) \quad \text{ if }1\leq c \leq k^-\\
  \\
  \frac{\alpha/m}{n-1+\alpha} f(y_i \vert \phi_c) \quad \text{ if }k^- < c \leq h \\
  \end{cases}
  \\
  \end{align*}
  $$
  
  * 적어도 한 개 이상의 observation과 연결이 된 $\phi_c$를 남기고 나머지는 버린다.

* For all $c \in \mathbf{c} = \{c_1, \cdots, c_n \}:$

  * 다음 분포로부터 $\phi_c$의 새 value를 생성한다.

  $$
  p(\text{d}\phi_c \vert \phi_{-c}, \mathbf{c}, y) \propto  \left[ \prod_{i: c_i = c} f(y_i \vert \phi_{c_i}) \right] G_0(\text{d} \phi_c)
  $$



### Derivation

위 알고리즘은 다음과 같은 conditional prior에서 도출된 알고리즘이다.

$$
\begin{align*}
p(c_i = c \vert \mathbf c_{-i},\phi_1, \cdots, \phi_h)
&\propto 
\begin{cases}
\frac{n_{-i,c}}{n-1+\alpha} \quad \text{ if }1\leq c \leq k^-\\
\\
\frac{\alpha/m}{n-1+\alpha} \quad \text{ if }k^- < c \leq h \\
\end{cases}
\\
\end{align*}
$$



## 6. Putting prior on concentration parameter $\alpha$ of DP prior

DP prior의 concentration parameter $\alpha$에 gamma prior를 부여한다.
