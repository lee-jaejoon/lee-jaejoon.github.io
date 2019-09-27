---
layout: post
title: "Measure-Theoretic Proof of Bayes' Theorem"
tags: [Statistics]
comments: true
---

이 포스트에서는 Bayes 정리의 증명을 목표로, 그를 위해 필요한 regular conditional distribution 개념을 소개한 후, Bayes' 정리의 증명과정을 소개한다.

# Regular Conditional Distribution

## Conditional Expectation / Probability

> **(Conditional expectation)** Given a probability space $(\Omega, \mathcal F, \mathbb P)$, let $\mathcal F_0 \subset \mathcal F $ be a sub $\sigma$-algebra of $\mathcal F $, and $X$ be a $\mathcal F$-measurable random variable with $\mathbb E\vert X\vert  < \infty$. The **conditional expectation** of $X$ given $\mathcal F_0$ is any real valued function $h : \Omega \rightarrow \mathbb R$, such that
>
> * $h$ is a $\mathcal F_0$-measurable random variable.
> * $\int_B h d\mathbb P = \int_B X d\mathbb P, \enspace \forall B \in \mathcal F_0$. 

또한 이러한 정의를 만족하는 $h$를 $h = \mathbb E[X \vert \mathcal F_0]$와 같이 나타낸다. 이러한 정의를 만족하는 conditional expectation은 유일하지 않을 수 있다. 따라서 그 중 한 function을 지칭할 때는 **"a version of"** conditional expectation of $X$ given $\mathcal F_0$라는 표현을 쓴다. 만약 $h_1$, $h_2$가 $\mathbb E[X \vert \mathcal F_0]$의 두 version이라고 하면 다음이 성립한다.

$$
\int_B h_1 d\mathbb P = \int_B h_2 d\mathbb P = \int X d\mathbb P, \enspace \forall B \in \mathcal F_0 \\
\iff h_1 = h_2 \text{ almost everywhere }[\mathbb P]
$$

따라서 conditional expectation의 version들은 모두 $[\mathbb P]$ almost everywhere 같은 random variable들이다.  
  


$\mathcal F $의 어떤 sub $\sigma$-algebra, $\mathcal F_0$에 대한 $A \in \mathcal F$의 **conditional probability**, $\text{Pr}(A \vert \mathcal F_0)$는 위의 conditional expectation의 정의에서 $X = 1_A$로 둠으로써 정의할 수 있다. 따라서 다음이 만족한다.

* $\text{Pr}(A \vert \mathcal F_0)$ is a $\mathcal F_0$-measurable random variable.
* $\int_B \text{Pr}(A \vert \mathcal F_0) d\mathbb P  = \mathbb P(A \cap B), \enspace \forall B \in \mathcal F_0$.

마찬가지로 conditional probability의 version들은 모두 $[\mathbb P]$ almost everywhere 같은 random variable들이다.  
   


## Regular Conditional Probability

위에서 정의한 대로, $\text{Pr}(A \vert \mathcal F_0)(\cdot)$은 $\Omega \rightarrow [0,1]$의 $\mathcal F_0$-measurable random variable이다. 이는 임의의 $A \in \mathcal F$에 대해 성립하는 statement이므로 우리는 $\text{Pr}(\cdot \vert \mathcal F_0)(\cdot)$를 $\mathcal F \times \Omega \rightarrow [0,1]$로 볼 수 있다. 또한 더 나아가, 임의의 주어진 $\omega \in \Omega $에 대해, $\text{Pr}( \cdot \vert \mathcal F_0)(\omega)$를 $(\Omega, \mathcal F)$ 위에서 정의된 probability measure로 보고자 한다. **Regular conditional probability**의 정의는 다음과 같다.

>  Assume $(\Omega, \mathcal F, \mathbb P) $ is a probability space. Let $\mathcal F_0 \subset \mathcal F$ be a sub $\sigma$-algebra. We say that function $\text{Pr}(\cdot \vert \mathcal F_0)(\cdot) : \mathcal F \times \Omega \rightarrow [0,1]$ is a **regular conditional probability** if
>
> * $\forall A \in \mathcal F, \text{Pr}(A\vert \mathcal F_0)(\cdot)$ is a version of $\mathbb E[1_A \vert \mathcal F_0]$.
> * For $[\mathbb P]$ almost everywhere $\omega \in \Omega$, $\text{Pr}(\cdot \vert \mathcal F_0)(\omega)$ is a probability measure on $(\Omega, \mathcal F)$.



**Regular conditional distribution**의 정의는 다음과 같다. 

> Assume $(\Omega, \mathcal F, \mathbb P) $ is a probability space. Let $X$ be a random variable such that $X:(\Omega, \mathcal F) \rightarrow (\mathfrak X, \mathcal X)$. For each $B \in \mathcal X$, let
> $$
> \mu_{X\vert \mathcal F_0}(B \vert \mathcal F_0)(\omega) = \text{Pr}(X^{-1}(B) \vert \mathcal F_0)(\omega).
> $$
> Then the function $\mu_{X\vert \mathcal F_0}(\cdot \vert \mathcal F_0)(\cdot) : \mathcal X \times \Omega \rightarrow [0,1]$ is called a **regular conditional distribution** of $X$ given $\mathcal F_0$ if
>
> * $\forall B \in \mathcal X, \mu_{X\vert \mathcal F_0}(B\vert \mathcal F_0)(\cdot)$ is a version of $\mathbb E[1_{ \{\omega : X(\omega) \in B \} } \vert \mathcal F_0]$.
> * For $[\mathbb P]$ almost everywhere $\omega \in \Omega$, $\mu_{X\vert \mathcal F_0}( \cdot \vert \mathcal F_0)(\omega)$ is a probability measure on $(\mathfrak X, \mathcal X)$.



## Existence of Regular Conditional Distribution

어떤 random variable $X:(\Omega, \mathcal F) \rightarrow (\mathfrak X, \mathcal X)$에 대해 regular conditional distribution이 존재하기 위해서는 어떤 조건이 필요할까? 다음 정리는 $\mathcal F$의 sub $\sigma$-algebra $\mathcal F_0$이 주어졌을 때 그에 대한 $X$의 regular conditional distribution이 존재하기 위한 조건을 제시한다.

> Regular conditional distributions exist if $(\mathfrak X, \mathcal X)$ is **nice**, i.e., there is a $1$-$1$ map $\varphi : \mathfrak X \rightarrow \mathbb R$ such that $\varphi$ and $\varphi^{-1}$ are measurable.

증명에 앞서 nice space의 예시를 몇 가지 소개한다.

* $(\mathfrak X, \mathcal X)$ where $\mathfrak X$ is any topological space and $\mathcal X$ is the Borel $\sigma$-field.
* $(\mathbb R^n, \mathcal B(\mathbb R^n))$
* The space $\mathcal C[0,1]$ of continuous functions on $[0,1]$ endowed with sup norm
* Polish spaces endowed with Borel $\sigma$-field.

**(증명 보충 要)**



## Motivation for the Definition of Regular Conditional Distribution

지금 이 포스트를 통해 conditional probability, regular conditional distribution의 개념을 처음 접했다면, regular conditional distribution의 정의에 등장하는 조건들이 무엇을 의미하는지 와닿지 않을 것이다 (적어도 이해가 빠른 편이 아닌 내겐 그랬다). 

Regular conditional distribution은 random variable $X$와 parameter random variable $\theta$ 사이의 관계를 나타내는 도구가 된다. 다음과 같은 예시 상황에의 적용을 통해 regular conditional distribution의 조건이 각각 어떤 의미를 갖는 것인지 알아보자.



다음과 같이 random variable $X, \theta$를 정의하자.

$$
X = \text{the number of heads(H) after tossing 3 identical coins simultaneously.}\\\theta = \text{the probability of heads(H)}
$$

우리는 "$\theta$가 주어졌을 때의 $X$의 conditional distribution"을 "$\sigma(\theta)$가 주어졌을 때 $X$의 **regular conditional distribution** $\mu_{X\vert\sigma(\theta)}(\cdot \vert \sigma(\theta))(\cdot)$"로 정의하며, statistical model을 $\mathcal P = \{ P_\theta : \theta \in \Theta \} = \{ \mu_{X\vert \theta}( \cdot \vert \theta) : \theta \in \Theta \}$로 정의한다. Regular conditional distribution $\mu_{X\vert\sigma(\theta)}(\cdot \vert \sigma(\theta))(\cdot)$는 그 정의에 의해 다음 조건들을 만족한다. 이 조건들이 각각 어떤 의미를 갖는지 하나하나 알아보자.

- 임의의 $B \in \mathcal X$에 대해, $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))(\cdot)$는 $\text{Pr}(X^{-1}(B) \vert \sigma(\theta))$의 한 version이다.
  - 임의의 $B \in \mathcal X$에 대해, $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))(\cdot)$는 $\sigma(\theta)$-measurable random variable이다.
  - 임의의 $B \in \mathcal X$에 대해, $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))(\cdot)$는 다음을 만족한다.

$$
\begin{align*}\int_C \mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta)) d\mathbb P &= \int_C \text{Pr}(X^{-1}(B) \vert\sigma(\theta)) d\mathbb P \\&=\int_C 1_{X^{-1}(B)} d\mathbb P \\&= \mathbb P \Big( X^{-1}(B) \cap C \Big), \enspace \forall C\in\sigma(\theta).\end{align*}
$$

-  $[\mathbb P]$ almost everywhere $\omega \in \Omega$에 대해, $\mu_{X\vert\sigma(\theta)}( \cdot \vert \sigma(\theta))(\omega)$는 $(\mathfrak X, \mathcal X)$에서 정의된 probability measure이다.



먼저 첫 번째 조건의 의미를 살펴보자.

> 임의의 $B \in \mathcal X$에 대해, $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))(\cdot)$는 $\sigma(\theta)$-measurable random variable이다.

$\theta$가 주어졌을 때의 $X$의 conditional distribution $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))$이 $\sigma(\theta)$-measurable random variable이라는 것은, 어떤 Borel function $g$에 대해 $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))$를 다음과 같이 random variable $\theta$의 함수로 나타낼 수 있다는 것과 동치이다. 자세한 내용이나 도출과정은 [factorization lemma](https://en.wikipedia.org/wiki/Factorization_lemma)를 참고하였다.

$$
\text{For }B \in \mathcal X, \enspace P_\theta( B ) = \mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta)) = g(\theta), \enspace \text{ for a Borel function } g.
$$

우리의 예시에 이를 적용하면 다음과 같다.

$$
\begin{align*}\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))(\omega) &= \Pr \Big( \Big\{ \omega : X(\omega) \in B \Big\} \Big\vert \sigma(\theta) \Big)(\omega) \\&= \sum_{x \in B} {3 \choose x} \theta(\omega)^x [1-\theta(\omega)]^{3-x} \\\end{align*}
$$

따라서 첫 번째 조건은 $\theta$가 주어졌을 때의 $X$의 conditional distribution이 $\sigma(\theta)$-measurable random variable, 즉 $\theta$에 대한 함수임을 의미한다.  
  


두 번째 조건의 의미를 살펴보자. 모든 $C\in\sigma(\theta)$는 그에 대해 $C = \theta^{-1}(D)$를 만족하는 $D \in \mathcal B(\Theta)$가 존재하므로, 두 번째 조건은 다음과 같이 다시 쓸 수 있다.

> 임의의 $B \in \mathcal X$에 대해, $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))(\cdot)$는 다음을 만족한다.
> 
> $$
> \begin{align*}\int_{ \theta^{-1}(D) } \mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta)) d\mathbb P &= \int_{ \theta^{-1}(D) } 1_{X^{-1}(B)} d\mathbb P \\&= \mathbb P \Big( X^{-1}(B) \cap { \theta^{-1}(D) } \Big) \\&=\mathbb P \Big( \Big\{ \omega : X(\omega) \in B, \theta(\omega) \in D \Big\} \Big), \enspace \forall D \in \mathcal B(\Theta).\end{align*}
> $$

그런데 우리는 $\mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))$가 random variable $\theta$에 대한 함수 $g(\theta)$라는 것을 위에서 확인하였다. 따라서 위 식의 좌변 $g(\theta) = \int_{ \theta^{-1}(D) } \mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta)) d\mathbb P $는 change-of-variable formula에 의해 다음과 같이 쓸 수 있다.

$$
\begin{align*}\int_{ \theta^{-1}(D) } \mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))(\omega) d\mathbb P(\omega)  &= \int_{ \theta^{-1}(D) } g(\theta(\omega)) d\mathbb P(\omega) \\&= \int_D g(\theta) d(\mathbb P \circ \theta^{-1})(\theta).\\\Big( &= \int_D g(\theta) \pi(\theta) d\eta(\theta). \Big)\\(\text{ if the prior distribution is domi}&\text{nated by some measure }\eta \text{. }) \end{align*}
$$

즉 다음이 만족한다.

$$
\int_D g(\theta) d(\mathbb P \circ \theta^{-1})(\theta) = \mathbb P \Big( \Big\{ \omega : X(\omega) \in B, \theta(\omega) \in D \Big\} \Big), \enspace \forall B \in \mathcal X, \forall D \in \mathcal B(\Theta)
$$

따라서 두 번째 조건은 $\theta$가 주어졌을 때의 $X$의 conditional distribution, $g(\theta) = \mu_{X\vert\sigma(\theta)}(B \vert \sigma(\theta))$에 prior density를 곱한 후, 이를 어떤 $D \in \Theta$ 위에서 $\theta$에 대해 적분하면 $X$, $\theta$의 joint probability가 도출된다는 (우리가 이미 당연하게 받아들이고 있는) 결과를 정당화한다.  
  


세 번째 조건의 의미를 살펴보자.

>  $[\mathbb P]$ almost everywhere $\omega \in \Omega$에 대해, $\mu_{X\vert\sigma(\theta)}( \cdot \vert \sigma(\theta))(\omega)$는 $(\mathfrak X, \mathcal X)$에서 정의된 probability measure이다.

여기서 중요한 점은 "어떤 $\omega_0 \in \Omega$로 $\omega$가 고정된다면, $\theta(\omega_0) \in \Theta $ 역시 고정된 값이 될 것"이라는 점이다. $\theta$가 주어졌을 때의 $X$의 conditional distribution이 $\theta$에 대한 함수라는 점을 이용하면, 임의의 $B \in \mathcal X$에 대하여,

$$
\begin{align*}
\Pr \Big( \Big\{ \omega : X(\omega) \in B \Big\} \Big\vert \sigma(\theta) \Big)(\omega) 
&= g\big(\theta(\omega) \big).
\end{align*}
$$

$B$와는 다른 $B^\prime \in \mathcal X$에 대해서도 conditional distribution은 $\theta$에 대한 함수일 것이지만, 이는 $g$와는 다른 $g^\prime$일 것이다.

$$
\begin{align*}
\Pr \Big( \Big\{ \omega : X(\omega) \in B^\prime \Big\} \Big\vert \sigma(\theta) \Big)(\omega)
&= g^\prime \big(\theta(\omega) \big).
\end{align*}
$$

$\theta(\omega_0)=0.52$라고 해보자.

$$
\begin{align*}
\Pr \Big( \Big\{ \omega : X(\omega) \in B \Big\} \Big\vert \sigma(\theta) \Big)(\omega_0)
&= \sum_{x \in B} {3 \choose x} \theta(\omega_0)^x [1-\theta(\omega_0)]^{3-x} \\
&= \sum_{x \in B} {3 \choose x} (0.52)^x (0.48)^{3-x} \\
&= G(B).
\end{align*}
$$

이는 $B \in \mathcal X $에 대한 set function으로 볼 수 있고, $[0,1]$의 값을 반환한다. 이 예시의 경우 $G:\mathcal X \rightarrow [0,1] $가 probability measure의 나머지 조건을 만족함을 쉽게 보일 수 있다. 즉 세 번째 조건은 주어진 $\omega \in \Omega $에 대해서, 또한 **그에 따라 주어진 $\theta \in \Theta$에 대해서**, $\theta$가 주어졌을 때의 $X$의 conditional distribution은 $X$에 대한 probability measure라는 것을 의미한다. 또한 이 결과는 $\mathbb P$에 대해 almost everywhere 성립한다.  
  
  

# Bayes' Theorem

Conditional distribution에 대한 엄밀한 정의와 regular conditional distribution의 개념에 대해 소개했다. 이를 바탕으로 Bayes' 정리에 대한 formal statement와 그 증명을 소개하며 이 포스트를 마친다.

## Statement

여기서는 증명 과정에서 notation의 정확성을 위해, parameter random variable을 $\Theta : (\Omega, \mathcal F) \rightarrow (\mathfrak Y, \mathcal Y ) $로 표기하고, 실현된 한 value를 $\theta$로 표기한다.

확률공간 $(\Omega, \mathcal F, \mathbb P)$에 대하여, random variable $X:(\Omega, \mathcal F) \rightarrow (\mathfrak X, \mathcal X)$와 $\Theta : (\Omega, \mathcal F) \rightarrow (\mathfrak Y, \mathcal Y )$를 생각해보자. 또한 $\Theta$가 주어졌을 때 $X$의 regular conditional distribution $\mu_{X\vert\Theta} (\cdot \vert \Theta)(\cdot)$이 존재한다고 가정하자.  즉 **$\Theta$가 주어졌을 때 $X$의 conditional distribution**이 잘 정의됨을 가정한다. (이 때 sub $\sigma$-algebra 자리에 random variable $\Theta $가 들어간 표현은 $\sigma(\Theta ) \subset \mathcal F$, 즉 random variable $ \Theta $를 measurable하게 하는 minimal $\sigma$-algebra를 대신하여 적은 것이다.) 또한 $P_\Theta$를 dominate하는 $\sigma$-finite measure $\nu$가 $(\mathfrak X, \mathcal X)$에 존재한다고 가정하자. 

$$
P_\Theta(B ) = \mu_{X\vert \Theta }(B \vert \Theta), \enspace \text{ for all }B \in \mathcal X, \text{ almost everywhere } [\mathbb P].\\
f_{X\vert \Theta}(x \vert \theta) = \frac{d \mu_{X\vert\Theta}(\cdot \vert \Theta)}{d \nu}, \text{ almost everywhere } [\mathbb P].
$$

우리는 $\mathcal P = \{ P_\Theta : \Theta \in \mathfrak Y \}$를 statistical model이라고 부르며, $\mathfrak Y $를 parameter space라고 부른다.

$$
\begin{align*}
\mathcal P &= \Big\{ \mu_{X\vert \Theta}( \cdot \vert \theta) : \theta \in \Theta \Big\} \\
&= \Big\{ \text{Pr}(X^{-1}(\cdot) \vert \theta) : \theta \in \Theta \Big\}
\end{align*}
$$

이 때 $f_{X\vert\Theta}(x \vert \theta ) $를 likelihood function이라고 하며, random variable $\Theta $의 distribution (induced probability measure) $\mu_\Theta $를 prior distribution이라고 한다. 이와 같은 setting에서 Bayes' 정리는 다음과 같다.

> **(Bayes' theorem)** Assume the structure above. Let $\mu_{\Theta\vert X}$ be the conditional distribution of $\Theta$ given $X$. Then the following holds.
>
> - $\mu_{\Theta \vert X} \ll \mu_\Theta$ almost everywhere with respect to the distribution of $X$.
> - $\frac{d \mu_{\Theta \vert X} }{d \mu_\Theta} = \frac{f_{X\vert\Theta}(x \vert \theta)}{ \int_{\mathfrak Y} f_{X \vert \Theta} (x \vert \theta) d\mu_\Theta(\theta)}$ for all $x$ for which the denominator is neither $0$ nor $\infty$.



## Proof

이 증명은 [Schervish - Theory of Statistics (1995)](https://www.springer.com/gp/book/9780387945460)의 Theorem 1.31의 내용을 옮긴 것이다.

먼저 분모에 대한 두 번째 statement를 먼저 증명하자. 다음과 같이 $C_0, C_\infty$를 정의한다.

$$
C_0 = \Big\{ x : \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta) = 0\Big\}, \\
C_\infty = \Big\{ x : \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta) d \mu_\Theta (\theta) = \infty \Big\}.
$$

$X$의 marginal distribution을 $\mu_X$라고 하자. $\mu_X$는 다음과 같이 도출할 수 있다.

$$
\begin{align*}
\mu_X(A) &= \mathbb P \Big( X^{-1}(A) \Big) \\
&=\int_\Omega \mu_{X\vert\Theta}(A \vert \Theta) d\mathbb P \\
\Big( \because \int_C &\mu_{X\vert\Theta}(A \vert \Theta) d\mathbb P =  \mathbb P \Big( X^{-1}(A) \cap C \Big), \enspace \forall C\in\sigma(\Theta) \Big) \\
&=\int_{\mathfrak Y} \mu_{X\vert\Theta}(A \vert \theta) d\mu_\Theta(\theta)\\
&=\int_{\mathfrak Y} \int_A\frac{ d\mu_{X\vert\Theta}( \cdot \vert \theta)}{d\nu} d\nu(x) d\mu_\Theta(\theta)\\
&=\int_A \int_{\mathfrak Y} \frac{ d\mu_{X\vert\Theta}( \cdot \vert \theta)}{d\nu} d\mu_\Theta(\theta) d\nu(x) \\
&=\int_A \int_{\mathfrak Y} f_{X \vert \Theta }(x \vert \theta) d \mu_\Theta(\theta) d\nu(x)\\
\end{align*}
$$

이를 이용하면,

$$
\begin{align*}
\mu_X(C_0) &=\int_{C_0} \int_{\mathfrak Y} f_{X \vert \Theta }(x \vert \theta) d \mu_\Theta(\theta) d\nu(x) =\int_{C_0} 0 \text{ } d\nu(x) = 0,\\
\mu_X(C_\infty) &=\int_{C_\infty} \int_{\mathfrak Y} f_{X \vert \Theta }(x \vert \theta) d \mu_\Theta(\theta) d\nu(x)=\int_{C_\infty} \infty \text{ } d\nu(x).
\end{align*}
$$

여기서 $\nu(C_\infty) >0$이면 $\mu_X(C_\infty) = \infty$이고, $\nu(C_\infty) = 0$이면 $\mu_X(C_\infty) = 0$임을 알 수 있다. $\mu_X( C_\infty ) \in [0,1] $이므로, 가능한 경우는 $\mu_X(C_\infty) = 0$이다. 따라서,

$$
\mu_X(C_0) = \mu_X(C_\infty) = 0.
$$



한편, posterior distribution $\mu_{\Theta \vert X}$는 그 정의에 따라 다음을 만족해야 한다.

$$
\begin{align*}
\int_B \mu_{\Theta \vert X}(D \vert x) d\mu_X (x) 
&= \int_{ X^{-1}(B) } \mu_{\Theta \vert X}(D \vert X) d\mathbb P \\
&= \int_{ X^{-1}(B) } 1_{\Theta^{-1}(D)} d\mathbb P \\
&= \mathbb P \Big( X^{-1}(B) \cap { \theta^{-1}(D) } \Big) \\
&=\mathbb P \Big( \Big\{ \omega : X(\omega) \in B, \theta(\omega) \in D \Big\} \Big) \\
&= \int_{ \Theta^{-1}(D) } 1_{X^{-1}(B)} d\mathbb P \\
&= \int_{ \Theta^{-1}(D) }  \mu_{X \vert \Theta}(B \vert \Theta) d\mathbb P  \\
&= \int_D \mu_{X \vert \Theta}(B \vert \theta) d\mu_\Theta (\theta) \\
&=\int_D \int_B \frac{d\mu_{X \vert \Theta}(\cdot \vert \theta)}{d\nu} d\nu(x) d\mu_\Theta (\theta) \\
&=\int_D \int_B f_{X \vert \Theta }(x \vert \theta)  d\nu(x) d\mu_\Theta (\theta) , \enspace \forall D \in \mathcal B(\Theta), \forall B \in \mathcal X.
\end{align*}
$$

따라서 Fubini 정리를 사용하면, $\forall D \in \mathcal B(\Theta), \forall B \in \mathcal X$에 대하여,

$$
\begin{align*}
\text{(a) }\int_B \mu_{\Theta \vert X}(D \vert x) d\mu_X (x)  
&= \int_D \int_B f_{X \vert \Theta }(x \vert \theta)  d\nu(x) d\mu_\Theta (\theta) \\
&= \int_B \int_D f_{X \vert \Theta }(x \vert \theta)   d\mu_\Theta (\theta)d\nu(x).
\end{align*}
$$


$\text{(a)}$ 결과를 기억해두고, $\int_B \mu_{\Theta \vert X}(D \vert x) d\mu_X (x)$를 다르게 전개해보자. 짚고넘어갈 사실은 $\mu_X  \ll \nu$라는 점이다. $\mu_{X\vert\Theta} \ll \nu$는 가정에 의해 주어졌으며, $\mu_X \ll \nu$는 다음과 같이 보일 수 있다. 임의의 $B \in \mathcal X$에 대해,

$$
\begin{align*}

\text{If }\nu(B) = 0, &\text{ then }\mu_{X \vert \Theta}(B \vert \Theta) = 0 \text{ a.e. }[\mathbb P], \\
 \text{ then, }\mu_X(B) &= \int_\Omega \mu_{X \vert \Theta}(B \vert \Theta) d \mathbb P \\
&=0.
\end{align*}
$$

따라서 우리는 $\nu$를 이용하여 $\mu_X$에 대해서도 Radon-Nikodym derivative를 정의할 수 있다. 이를 이용하여 적분식을 정리하면, $\forall D \in \mathcal B(\Theta), \forall B \in \mathcal X$에 대하여,

$$
\begin{align*}
\text{(b) }\int_B \mu_{\Theta \vert X}(D \vert x) d\mu_X (x)  
&= \int_B  \mu_{\Theta \vert X}(D \vert x) \left( \frac{d\mu_X}{d\nu}(x) \right) d\nu (x) \\
&= \int_B \mu_{\Theta \vert X}(D \vert x) \left( \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta) \right) d\nu (x).
\end{align*}
$$

두 번째 등호가 성립하는 것은, Fubini 정리를 사용하여 $\mu_X$의 Radon-Nikodym derivative를 다음과 같이 나타낼 수 있기 때문이다.

$$
\begin{align*}
\int_B \left( \int_{\mathfrak Y}\frac{d \mu_{X\vert\Theta}( \cdot \vert \theta)}{d\nu}d\mu_\Theta(\theta) \right) d\nu(x) &= \int_{\mathfrak Y} \int_B  \frac{d \mu_{X\vert\Theta}( \cdot \vert \theta)}{d\nu} d\nu(x) d\mu_\Theta(\theta) \\
&=  \int_{\mathfrak Y} \mu_{X\vert\Theta}( B \vert \theta) d\mu_\Theta(\theta) \\
&= \mu_X(B)  , \enspace \forall B \in \mathcal X.
\end{align*}
$$

$$
\therefore \enspace \frac{d\mu_X}{d\nu} = \frac{d\Big(\int_{\mathfrak Y} \mu_{X\vert\Theta}( \cdot \vert \theta) d\mu_\Theta(\theta) \Big)}{d\nu} = \int_{\mathfrak Y}\frac{d \mu_{X\vert\Theta}( \cdot \vert \theta)}{d\nu}d\mu_\Theta(\theta).
$$



$\text{(a),(b)}$의 두 결과를 종합하면, $ \forall D \in \mathcal B(\Theta), \forall B \in \mathcal X$에 대하여,

$$
\int_B \int_D f_{X \vert \Theta }(x \vert \theta)   d\mu_\Theta (\theta)d\nu(x) =\int_B  \left(\mu_{\Theta \vert X}(D \vert x) \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta) \right) d\nu (x)
$$

$$
\begin{align*}
&\Rightarrow \int_D f_{X \vert \Theta }(x \vert \theta)   d\mu_\Theta (\theta) = \mu_{\Theta \vert X}(D \vert x) \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta) , \quad \text{a.e. }[\nu].\\
&\Rightarrow \mu_{\Theta \vert X}(D \vert x) = \frac{\int_D f_{X \vert \Theta }(x \vert \theta)d \mu_\Theta (\theta)}{ \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta)} , \quad \text{a.e. }[\nu].\\
&\Rightarrow \mu_{\Theta \vert X}(D \vert x) = \frac{\int_D f_{X \vert \Theta }(x \vert \theta)d \mu_\Theta (\theta)}{ \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta)} , \quad \text{a.e. }[\mu_X]. \quad(\because \text{ }\mu_X \ll \nu)
\end{align*}
$$

따라서 임의의 $D \in \mathcal B(\Theta)$에 대해, $\mu_\Theta(D)=0 \implies  \mu_{\Theta \vert X}(D \vert x)=0$이 성립한다. 즉 $\mu_{\Theta \vert X} \ll \mu_\Theta $이 성립한다.  
  
또한 다음과 같이 $\mu_\Theta$에 대한 $\mu_{\Theta \vert X}$의 Radon-Nikodym derivative를 구할 수 있다.

$$
\int_D \left(  \frac{f_{X \vert \Theta }(x \vert \theta)}{ \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta)} \right)d \mu_\Theta (\theta) = \mu_{\Theta \vert X}(D \vert x) \\
\implies \frac{d\mu_{\Theta \vert X}}{d\mu_\Theta} = \frac{f_{X \vert \Theta }(x \vert \theta)}{ \int_{\mathfrak Y} f_{X \vert \Theta}(x \vert \theta ) d \mu_\Theta (\theta)}. 
$$

이로써 Bayes' Theorem의 모든 statement이 증명되었다.$\quad \quad \square$
