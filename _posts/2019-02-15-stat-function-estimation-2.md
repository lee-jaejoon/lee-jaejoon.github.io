---
layout: post
title: "Statistical Learning as Function Approximation: 2. Reproducing Kernel Hilbert Space"
tags: [Statistics]
comments: true
---

Reproducing kernel Hilbert space는 **함수를 어떤 집합 내의 객체로 바라봄**으로써, 모형의 학습을 함수 추정의 문제로 바라보는 다른 시각을 제시해준다. 이 포스트는 Reproducing kernel Hilbert space의 기본적인 개념들에 대해 소개하는 것을 목표로 한다. 아래에서 언급한 대부분의 정의는 Wikipedia의 정의를 사용하였다.

# Necessary Concepts

### Definition: Real Inner Product Space

 > A **real inner product space** is a vector space $V$ over the field $\mathbb{R}$ together with an *inner product*, i.e., with a map
 > 
 > $$
 > \langle \cdot ,\cdot  \rangle : V \times V \rightarrow \mathbb{R}
 > $$
 > 
 > that satisfies the following three axioms for all vectors $x,y,z \in V$ and all scalars $a \in \mathbb{R}$
 > 1. Symmetry : $\langle x ,y  \rangle=\langle y,x  \rangle$
 > 2. Bilinearity : $\langle ax+y ,z  \rangle=a\langle x ,z  \rangle +\langle y,z  \rangle$
 > 3. Positive definiteness : $\langle x ,x  \rangle \ge0 , \enspace \langle x ,x  \rangle=0 \iff x=\mathbf{0}$

Real inner product space는 위 세 가지 성질을 만족하는 mapping, $\langle \cdot ,\cdot  \rangle : V \times V \rightarrow \mathbb{R}$, 즉 inner product가 정의된 $\mathbb{R}$-vector space를 의미한다.

### Definition: Hilbert Space

 > A **Hilbert space**, $H$ is a real or complex inner product space that is also a complete metric space with respect to the distance function induced by the inner product.

Hilbert space란 complete한 inner product space를 의미한다. 임의의 inner product에 대해 우리는 inner product의 square root로 norm을 정의할 수 있기 때문에 inner product space는 normed space이다.

$$
||x||= \langle x,x \rangle ^{\frac{1}{2}}
$$

또한, norm이 존재하면 두 원소 간의 distance(metric)를 다음과 같이 정의할 수 있기 때문에, 당연하게 metric space가 된다. 

$$
d(x,y)=||x-y||
$$

어떤 metric space가 complete하다는 것은 metric space 내의 임의의 Cauchy 수열이 그 metric space 내의 원소로 수렴한다는 것을 의미한다. 여기서 우리는 complete **real** inner product space로서의 Hilbert space만을 고려할 것이다.


### Definition: Evaluation Functional

 > Let $\mathcal{X}$ be an arbitrary set and $\mathcal{H}$ a Hilbert space of real-valued functions on $\mathcal{X}$. The **evaluation functional** over the Hilbert space of functions $\mathcal{H}$ is a linear functional that evaluates at each function at a point $x$.
 > 
 > $$
 > L_x:f \mapsto f(x) \enspace \forall f \in \mathcal{H}
 > $$
 > 

엄밀한 정의는 아니지만 functional이란 흔히 "function of a function", 즉 함수를 input으로 받는 함수를 의미한다. Evaluation은 "어떤 점에서의 함수가 어떤 값을 갖는지 evaluate하는 것", 다시 말해서 함수 $f$에 $x$를 대입하여 그 때의 값 $f(x)$를 얻는 것을 의미한다. Evaluation functional은 이 함수의 evaluation을 수행해주는 functional이다. $f$에 $x$를 대입하여 함숫값 $f(x)$를 얻는 그 행위를, 다른 관점에서 $f$를 evaluation functional $L_x$에 넣는 행위로 보는 것이다.

$$
L_x(f)=f(x)
$$

말장난 같지만, 이는 **함수를 객체로 바라보는** 관점에서 쓰여진 Reproducing Kernel Hilbert Space를 이해하는데 필수적이다. 아래의 설명에서 필요한 evaluation functional의 성질을 미리 소개하겠다.

 > Evaluation functional $L_x$ is **linear** functional of $\mathcal H$, i.e., the following satisfies for all $f,g \in \mathcal H$ and for all $a \in \mathbb{R}$.
 > 
 > $$
 > L_x(af+g)=a\cdot L_x(f) + L_x(g)
 > $$
 >

바로 Evaluation functional은 linear하다는 성질이다. 이는 다음과 같이 쉽게 보일 수 있다.

$$
L_x(af+g)=(af+g)(x)=a \cdot f(x)+g(x)=a\cdot L_x(f) + L_x(g)
$$


### Definition: Topological Dual Space

 > Let $H$ be a topological vector space over the field $F$. A corresponding **topological dual space**, $H^\ast$ is a space consisting all continuous linear functionals on $H$.
 > 
 > $$
 > H^\ast = \{ L : H \rightarrow F \mid L \text{ is linear and continuous}\}
 > $$
 > 

임의의 topological vector space $H$에 대하여, 그와 짝을 이루는 topological dual space $H^\ast$가 존재한다. 이 topological dual space는 함수들의 공간인데, $H$를 $F$로 보내는 모든 linear function을 모아놓은 공간(혹은 집합)이다. 만약 우리의 vector space $H$가 실수 위에서 정의되었다면, $H^\ast$는 $H$에서 $\mathbb{R}$로 가는 모든 linear function을 모아놓은 집합이다. 연속조건이 없이 모든 linear functional들을 모은 일반적인 dual space(algebraic dual space)를 사용하지 않는 이유는, 우리의 관심 vector space는 함수를 그 원소로 갖는 topological vector space이기 때문이다.

# Reproducing Kernel Hilbert Space
## 1. Definition

Reproducing kernel Hilbert space는 다음과 같이 정의된다.

 > We say that a Hilbert space of real-valued function of $\mathcal X$, $\mathcal H$ is a **reproducing kernel Hilbert space** if, for all $x \in \mathcal X$, $L_x$ is continuous at any $f \in \mathcal H$ or equivalently, if $L_x$ is a bounded operator on $\mathcal H$ , i.e. there exists an $M$ such that,
 > 
 >  $$
 >  |L_x(f)| = |f(x)| \le M ||f||_\mathcal H \enspace \forall f \in \mathcal H
 >  $$
 >  

한국어로 직역해보겠다.

 > $\mathcal H$는 $\mathcal X$의 원소를 넣으면 실수 값을 반환하는 함수들($?:\mathcal X \rightarrow \mathbb{R}$)의 Hilbert space이다. 다음 성질이 만족할 때, 우리는 $\mathcal H$를 reproducing kernel Hilbert space라고 부른다. 다음 두 성질은 서로 동치이다.
 >  * $\mathcal X$의 임의의 원소 $x$에 대하여, evaluation functional $L_x$가 $\mathcal H$의 임의의 점(그렇지만 함수인) $f$에서 연속이다. 즉, 임의의 $x$에 대하여, evaluation functional $L_x$가 $\mathcal H$에서 연속함수이다.
 >  * $\mathcal X$의 임의의 원소 $x$에 대하여, evaluation functional $L_x$가 $\mathcal H$에서 bounded operator이다.
 >  
 >  $$
 >  |L_x(f)| = |f(x)| \le M ||f||_\mathcal H \enspace \forall f \in \mathcal H
 >  $$
 >  

어렵다. 이 정의를 읽고 "아, RKHS가 이런거구나"하고 이해할 수 있는 사람은 거의 없을 것이다. reproducing kernel Hilbert space는 이 정의로부터 얻을 수 있는 다음 theorem을 통해 이해할 수 있다.

## 2. Riesz Representation Theorem

먼저 Riesz representation theorem의 의미를 충분히 설명한 다음, 이 theorem의 증명을 소개하겠다. 

### Statement
Riesz representation theorem의 statement는 다음과 같다.

 > Let $\mathcal H$ be a Hilbert space over $\mathbb{R}$. If $T \in \mathcal H^\ast$, then there exists a unique vector $u$ in $\mathcal H$ such that
 > 
 > $$
 > T(v)=\langle v,u \rangle_\mathcal H \text{ for all } v \in \mathcal H
 > $$


우선 statement를 이해하기 위해, 한국어로 직역해보겠다.

 > $\mathbb{R}$ 위에서 정의된 Hilbert space $\mathcal H$가 있다고 하자. 만약 $T$가 $\mathcal H^\ast$의 원소라면, $\mathcal H$ 안의 임의의 원소 $v$에 대해 $T(v)=\langle v,u \rangle_\mathcal H$를 만족하는 $u \in \mathcal H$가 유일하게 존재한다.

$\mathbb{R}$ 위에서 정의된 Hilbert space $\mathcal H$가 있을 때, $T$가 $\mathcal H$의 topological dual space $\mathcal H^\ast$의 원소라는 것은 $T$가 $\mathcal H$에서 $\mathbb{R}$로 가는 continuous linear function이라는 뜻이다.

$$
T : \mathcal H \rightarrow \mathbb{R} \text{ , and } T \text{ is linear and continuous function}
$$

$T : \mathcal H \rightarrow \mathbb{R} $인 임의의 linear continuous function $T$에 대해, $\mathcal H$ 안의 어떤 $u$가 **유일하게** 존재해서 그냥 $T$라는 linear continuous function을 다음과 같이 나타낼(represent) 수 있다고 한다.

$$
T(v)=\langle v,u \rangle_\mathcal H \text{ for all } v \in \mathcal H
$$

즉, $\mathcal H$에서 $\mathbb{R}$로 가는 어떤 linear continuous function을 잡더라도, 그 **"linear continuous function에 input $v$를 넣는 것"**은 **"$v$를 Hilbert space $\mathcal H$ 내의 한 벡터 $u$와 내적시키는 것"**과 같다는 것이다. 그렇기 때문에 임의의 linear continuous function $T : \mathcal H \rightarrow \mathbb{R} $를 $T(v)=\langle v,u \rangle_\mathcal H$의 형태로 **나타낼 수 있다**, **represent할 수 있다**는 것이 Riesz representation theorem의 의미이다.  


### Proof

### Existence

만약 $T$가 zero functional이라면, $u=\mathbf{0}$일 때 위 theorem이 성립한다.

$$
T(v)=\langle v,\mathbf{0} \rangle_\mathcal H =0\text{ for all } v \in \mathcal H
$$

만약 $T$가 zero functional이 아닐 때를 고려해보자. 임의의 $w \in \text{ker}(T)^\perp$에 대해, 다음과 같이 $u \in \text{ker}(T)$를 정의한다.

$$
u = \frac{T(w)}{||w||^2_\mathcal H} w
$$

이 때, 다음과 같은 식이 성립한다.

$$
\langle w,u \rangle_\mathcal H = \left\langle w, \frac{T(w)}{||w||^2_\mathcal H} w \right\rangle_\mathcal H = \frac{T(w)}{||w||^2_\mathcal H} \langle w, w \rangle_\mathcal H = T(w)
$$

그러므로, $T(w)=\langle w,u \rangle_\mathcal H , \text{ for all } w \in \text{ker}(T)^\perp$가 성립한다.

으아아아아ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ

### Uniqueness

으아아아아ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ  

<br>
## 3. Reproducing Property: Applying Riesz represnetation theorem to RKHS

그럼 Riesz representation theorem은 reproducing kernel Hilbert space에서 어떤 의미를 가질까?  

Reproducing kernel Hilbert space의 정의에서, 우리는 $\mathcal H$를 a Hilbert space of real-valued function, 즉 실수 값을 반환하는 함수의 Hilbert space로 보았다. 즉 reproducing kernel Hilbert space를 생각할 때는 Hilbert space $\mathcal H$의 원소가 함수라는 것을 항상 생각해야 한다.(그렇지 않으면 절대로 이해할 수 없다.) 잊지 말자. 우리는 함수를 객체로 생각한다 $\cdots$ $\mathcal H$의 원소들은 각각 다 함수다 $\cdots$ 우리는 함수를 객체로 생각한다 $\cdots$ $\mathcal H$의 원소들은 각각 다 함수다 $\cdots \cdots$  
  
위에서 소개한 evaluation functional을 생각해보자. Evaluation functional $L_x$는 $f \in \mathcal H$를 받아 $f(x) \in \mathbb{R}$를 반환하는 함수였다. 그리고 우리는 evaluation functional $L_x$가 linear하다는 것을 위에서 확인하였다. 또한 reproducing kernel Hilbert space의 정의에 따라 $\mathcal H$에서 정의된 evaluation functional은 continuous하다. 따라서, 

$$
L_x : \mathcal H \rightarrow \mathbb{R} \text{ , and } L_x \text{ is linear and continuous function}
$$

이제 Riesz representation theorem에 의해, $L_x$에 대해 다음을 만족시키는 $K_x \in \mathcal H$가 **유일하게** 존재할 것이다.

$$
L_x(f)=\langle f,K_x \rangle_\mathcal H \text{ for all } f \in \mathcal H
$$

"evaluation functional $L_x$에 input $f$를 넣는 것"은 "$f$를 Hilbert space $\mathcal H$ 내의 한 원소 $K_x$와 내적시키는 것"과 같다. 그런데 evaluation functional의 정의에 의하면, "evaluation functional $L_x$에 input $f$를 넣는 것"은 "함수 $f$에 $x$를 대입하는 것"과 같다. 따라서, **"함수 $f$에 $x$를 대입하는 것"**이 **"Hilbert space $\mathcal H$의 원소인 $f$를 다른 한 원소 $K_x$와 내적시키는 것"**과 같다는 것이다. 물론 다시 강조하지만, $\mathcal H$의 원소들은 각각 다 함수이기 때문에 $f$와 $K_x$는 $\mathcal X \rightarrow \mathbb{R}$인 함수이다. 이를 **Reproducing property**라고 한다.

$$
f(x)=\langle f,K_x \rangle_\mathcal H \text{ for all } f \in \mathcal H
$$

함수에 어떤 값을 대입하는 것(evaluation)을, 그 함수가 객체로 존재하는 공간($\mathcal H$)에서 다른 어떤 적당한 함수($K_x$)와 내적하는 것으로 바라볼 수 있게 되었다. 덧붙이자면, 여기서 $K_x$를 쓸 때 아래 첨자로 $x$를 붙인 이유는, $x \in \mathcal X$가 달라지면 evaluation functional $L_x: \mathcal H \rightarrow \mathbb{R}$도 달라지고 그에 따라 $K_x \in \mathcal H$도 달라지기 때문이다. 예를 들어, $x$가 아닌 다른 input $x' \in \mathcal X$를 evaluate하는 것은 $K_x$가 아닌 $K_{x'}$와 연결될 것이다.

## 4. Reproducing Kernel

위에서 도출한 reproducing property을 이용하여 reproducing kernel이 무엇인지 도출해보자. $\mathcal X$의 임의의 두 원소 $x,y$에 대해, reproducing property로 정의된 $K_x,K_y \in \mathcal H$를 다음과 같이 나타낼 수 있다.

$$
L_x(f)=\langle f,K_x \rangle_\mathcal H =f(x) \text{ for all } f \in \mathcal H
$$

$$
L_y(f)=\langle f,K_y \rangle_\mathcal H =f(y) \text{ for all } f \in \mathcal H
$$

그런데 $K_x,K_y$도 $X$의 원소를 $\mathbb{R}$로 보내는 함수, 즉 $H$의 원소이다. 따라서 다음과 같이 쓸 수 있다.

$$
L_x(K_y)=\langle K_y,K_x \rangle_\mathcal H =K_y(x) 
$$

$$
L_y(K_x)=\langle K_x,K_y \rangle_\mathcal H =K_x(y)
$$

그런데 inner product $\langle \cdot , \cdot \rangle$는 symmetric한 함수이므로, $\langle K_y,K_x \rangle_\mathcal H=\langle K_x,K_y \rangle_\mathcal H$가 만족한다. 따라서 우리는 **reproducing kernel**, $K:\mathcal X \times \mathcal X \rightarrow \mathbb{R}$을 다음과 같이 정의한다.

$$
K(x,y)=\langle K_x,K_y \rangle_\mathcal H =K_y(x)=K_x(y)
$$

### Properties of Reproducing Kernel

위와 같이 정의된 reproducing kernel $K:\mathcal X \times \mathcal X \rightarrow \mathbb{R}$는 symmetric하고 positive definite인 성질이 있다.
### Symmetry

$$
K(x,y)=\langle K_x,K_y \rangle_\mathcal H =\langle K_y,K_x \rangle_\mathcal H=K(y,x)
$$

### Positive-definiteness

For any $n \in \mathbb{N}$, $x_1,\cdots,x_n \in \mathcal X$, $c_1, \cdots, c_n \in \mathbb{R}$,

$$
\sum_{i=1}^{n} \sum_{j=1}^{n} c_i c_j K(x_i, x_j) \ge 0
$$

이는 다음과 같이 쉽게 보일 수 있다.

$$
\sum_{i=1}^{n} \sum_{j=1}^{n} c_i c_j K(x_i, x_j) =\sum_{i=1}^{n} \sum_{j=1}^{n} c_i c_j \langle K_{x_i},K_{x_j} \rangle_\mathcal H =\left\langle  \sum_{i=1}^{n} c_i K_{x_i}, \sum_{j=1}^{n}  c_j  K_{x_j} \right\rangle_\mathcal H  \ge 0
$$

## 5. Moore–Aronszajn theorem

지금까지 우리는 어떤 reproducing kernel Hilbert space $\mathcal H$가 있을 때, 그로부터 symmetric하고 positive definite인 reproducing kernel $K$를 **유일하게** 정의하는 과정을 살펴보았다. 

$$
\text{Reproducing kernel Hilbert space: }\mathcal H \enspace 
\xrightarrow{\text{Riesz representation thm}}
\enspace \text{Symmetric, positive-definite kernel: }K
$$

Moore-Aronszajn theorem은 그와 정반대의 방향의 논리를 전개한다. Symmetric, positive definite를 만족하는 임의의 kernel $K$가 reproducing kernel Hilbert space $H$를 **유일하게** 정의한다.


$$
\text{Reproducing kernel Hilbert space: }\mathcal H \enspace 
\xleftarrow{\text{ Moore–Aronszajn thm }}
\enspace \text{Symmetric, positive-definite kernel: }K
$$

Moore-Arnoszajn theorem의 statement는 다음과 같다.

 > Suppose $K$ is a symmetric, positive definite kernel on a set $\mathcal X$. Then there is a **unique** Hilbert space $\mathcal H_K$ of functions on $\mathcal X$ for which $K$ is a reproducing kernel.  
 >  
 > 집합 $\mathcal X$에서 정의된 symmetric, positive definite kernel $K$에 대해, $\mathcal X$의 원소를 넣으면 실수 값을 반환하는 함수들의 Hilbert space이고, $K$를 reproducing kernel로 갖는 $\mathcal H_K$가 **유일하게** 정의된다.

### Proof
### Existence
Symmetric, positive definite를 만족하는 kernel $K$가 주어졌다고 가정하자. 이를 통해, 모든 $x \in \mathcal X$에 대해, 함수 $K_x: \mathcal X \rightarrow \mathbb{R}$를 다음과 같이 정의하자.

$$
K_x(\cdot)=K(x, \cdot)
$$

$\\{ K_x : x \in \mathcal X \\}$의 linear span을 $\mathcal H_0$라고 하자. $\mathcal H_0$의 임의의 두 원소 $g_1, g_2$는 다음과 같이 나타낼 수 있다. $g_1, g_2$는 $\mathcal X \rightarrow \mathbb{R}$인 함수들을 선형결합한 것이므로 마찬가지로 $\mathcal X \rightarrow \mathbb{R}$인 함수이다.

$$
g_1= \sum_{i=1}^{m} \alpha_i K_{x_i} \text{ , }  g_2= \sum_{j=1}^{n} \beta_j K_{x'_j}
$$

Symmetric, positive definite kernel $K$를 이용하여, 집합 $\mathcal H_0$의 inner product를 다음과 같이 정의할 수 있다.

$$
\langle g_1, g_2 \rangle_{\mathcal H_0}= \sum_{i=1}^{m} \sum_{j=1}^{n} \alpha_i \beta_j  K(x_i , x'_j)
$$

$K$가 Symmetric, positive definite kernel이기 때문에, $\langle \cdot , \cdot \rangle_{\mathcal H_0}$가 inner product의 조건을 만족한다는 것은 어렵지 않게 확인할 수 있다. $\mathcal H$가 $\mathcal H_0$의 inner product로 도출된 **[completion](https://en.wikipedia.org/wiki/Complete_metric_space#Completion)**이라고 하자. $\mathcal H$는 inner product space의 completion이므로, complete inner product space, 즉 Hilbert space가 된다. 이 때 $\mathcal H$의 원소는 다음과 같이 나타낼 수 있다. $\langle f,f \rangle_\mathcal H < \infty$은 Cauchy-Schwartz 부등식을 사용하여 보일 수 있다.

$$
f =  \sum_{i=1}^{\infty} \alpha_i K_{x_i} \text{ , } \enspace \text{ where } \langle f,f \rangle_\mathcal H =\sum_{i=1}^{\infty} \alpha_i^2 K(x_i,x_i)< \infty
$$

이와 같이 정의된 공간 $\mathcal H$가 reproducing property를 만족하는지 확인해보자. 마지막 등식은 $K_{x_i}(\cdot)=K(x_i, \cdot)$ 때문에 만족한다.

$$
\langle f,K_x \rangle_\mathcal H = \left\langle \sum_{i=1}^{\infty} \alpha_i K_{x_i} ,K_x \right\rangle_\mathcal H =\sum_{i=1}^{\infty} \alpha_i   K(x_i ,x)=f(x)
$$

따라서, Symmetric, positive definite를 만족하는 kernel $K$가 주어졌을 때, $K$를 reproducing kernel로 갖는 reproducing kernel Hilbert space $\mathcal H$를 만들 수 있다.

### Uniqueness

$\mathcal G$가 $K$를 reproducing kernel로 갖는 또다른 Hilbert space라고 하자. 임의의 $x,y \in \mathcal X$에 대해, reproducing property에 의해 다음 식이 만족한다.

$$
\langle K_x , K_y \rangle_\mathcal H =K(x,y) =\langle K_x , K_y \rangle_\mathcal G
$$

으아ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ



## 6. Examples of Positive-definite Kernels

 * Linear kernel

$$
K(x,y)=x^T y
$$

 * Gaussian kernel

$$
K(x,y)=\exp \Big( \frac{||x-y||^2}{\sigma^2} \Big) , \enspace \sigma > 0
$$

 * Polynomial kernel

$$
K(x,y)=(x^Ty +1)^d  , \enspace d \in \mathbb{N}
$$

<br>
## 7. Integral Operator
## 8. Mercer Theorem
## 9. Feature Maps

# Summary

![image](https://user-images.githubusercontent.com/45325895/52788494-79ef3380-30a4-11e9-9338-a4e22dcb0e76.png){: .center-image}

Symmetric, positive definite인 kernel $K: \mathcal X \times \mathcal X \rightarrow \mathbb{R}$는 $\mathcal X$의 두 원소($x,y$)를 input으로 받아 실수($K(x,y)$)를 함숫값으로 반환하는 함수이다. Symmetric, positive definite kernel은 어떤 적절한 vector space의 두 원소를 내적하는 것으로도 볼 수 있다. 이 적절한 vector space를 reproducing kernel Hilbert space라고 부른다. 놀라운 점은, 임의의 reproducing kernel Hilbert space는 symmetric, positive-definite한 reproducing kernel을 유일하게 정의하고, 반대로 임의의 symmetric, positive-definite kernel은 고유한 reproducing kernel Hilbert space를 정의한다는 것이다.

<br>

# Reference
 > Gockenbach, M. S. (2010). Finite-dimensional linear algebra. CRC Press.
 > https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space  
 > http://www.mit.edu/~9.520/spring06/Classes/class03.pdf




