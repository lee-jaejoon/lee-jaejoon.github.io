---
layout: post
title: "1. Graphical Models"
tags: [Probabilistic Graphical Model]
comments: true
---

이 포스트는 Michael I. Jordan, *"An Introduction to Probabilistic Graphical Models"*, 2000의 Chapter 1, Graphical Models를 정리한 것이다.

# Preliminaries

* *Walk*는 node들의 sequence들을 잇는 edge들의 sequence이다. 
  * Infinite이어도 됨.
* *Trail*은 모든 edge들이 distinct한 walk이다. 
  * 즉 edge의 sequence에서 같은 edge가 두 번 이상 등장하면 안됨.
* *Path*는 등장하는 모든 node들이 distinct한 walk이다. 
  * Node들이 distinct하면 그에 따라 edge들도 distinct하므로 path는 항상 trail이다.
* *Cycle*은 첫 번째 node와 마지막 node만이 서로 같은 node이고, 나머지 node들은 한 번 씩만 등장하는 non-empty trail이다.
* *Chord*는 cycle의 일부가 아니지만 cycle에 등장하는 두 node들을 잇는 edge이다.
* *Induced subgraph of* $G$는, 어떤 graph $G$의 node들의 부분집합에 대해, 그 부분집합에 속하는 node들 그리고 그 node들을 잇는 모든 edge로 생성된 graph이다.
* *Induced path in* $G$는 induced subgraph of $G$인 path이다.
* *Induced cycle in* $G$는 induced subgraph of $G$인 cycle이다.
  * *Chordless cycle*, 혹은 *hole*로도 불린다.
* *Chordal graph*는 graph 내의 4개 이상의 node로 이루어진 모든 cycle은 항상 chord를 갖는 graph를 의미한다.
  * Chordal graph의 모든 induced cycle은 3개의 node를 가져야 하며, 이를 통해 chordal graph를 정의하기도 한다.

<br>

# Chapter 1. Graphical Models

# 1.1. Introduction

* *Graphical model*은 graph로 정의된 확률분포의 family를 의미한다. 

* 이 때 graph의 node는 확률변수가 되고, 연결된 node들의 부분집합 위에서 정의된 함수들의 곱으로 결합확률분포를 정의한다. 
* Graphical model의 formalism은 빈도주의자 혹은 베이지안 중 어느 관점에 근거한 것인지 정확히 분별하기 힘들지만, 결합확률분포를 조작하는 방법이나 계층적 잠재 변수 모형을 쉽게 반영할 수 있다는 점에서, graphical model은 주로 베이지안 체계에서 논의/서술된다.
* Graphical model을 이용해 여러 분야의 복잡한 현상을 나타내는 probabilistic model을 효과적으로 만들 수 있다.

<br>



# 1.2 Representation

* Graphical model은 크게 *directed graphical model*과 *undirected graphical model*로 나눌 수 있다.

<br>

## 1.2.1 Directed case

* $\mathcal G(\mathcal V, \mathcal E)$ : directed acyclic graph, where $\mathcal V$ are nodes and $\mathcal E$ are the edges.
* $\{ X_v :v \in \mathcal V \}$ : a collection of random variables indexed by the nodes of the graph.
* 각 node $v \in \mathcal V$에 대해, $\pi_v$ : the subset of indices of its parents.
* 결합 확률 분포 $p(x_\mathcal V)$은 $\{ k(x_v \vert x_{\pi_v}) \}$들의 곱으로 나타낸다.

$$
p(x_\mathcal V) = \prod_{v \in \mathcal V} k (x_v \vert x_{\pi_v})
$$

* $p(x_v \vert \cdots ) = p(x_v \vert x_{\pi_v})$이므로, kernel을 그냥 conditional로 나타낸다.

$$
k(x_v \vert x_{\pi_v}) = p(x_v \vert x_{\pi_v})
$$

* 지금 여기서 data와 parameter를 구분하고 있지 않은데, parameter도 하나의 node로 모형에 반영시켜주면 된다.

<br>

## 1.2.2 Undirected case

* $\mathcal G(\mathcal V, \mathcal E)$ : undirected acyclic graph.
* $\{ X_v :v \in \mathcal V \}$ : a collection of random variables indexed by the nodes of the graph.

* $\mathcal C$ : *cliques* of the graph. Node들의 fully-connected subset들을 의미. 모두 이어진 덩어리들.

* 각 clique $C \in \mathcal C$에 대해, $\psi_C(x_C)$ : a nonnegative *potential* function.
* 결합 확률 분포 $p(x_\mathcal V)$은 potential function $\{\psi_C(x_C) \}$들의 곱의 normalized version으로 나타낸다.

$$
p(x_\mathcal V) = \frac{1}{Z} \prod_{C \in \mathcal C} \psi_C(x_C)
$$



* 주로 공간 통계학이나, 자연어 처리, 네트워크 데이터에 사용된다.

* Clique $C$가 매우 큰 경우가 있기 때문에, clique 내에 factorized distribution의 형태를 가져 그 역시 어떤 다른 함수들의 곱으로 나타나는 potential function을 고려하는 것이 효과적이다.

<br>



# 1.3 Algorithms for probabilistic inference

* Inference를 수행할 때는 directed graph와 undirected graph를 같은 방법으로 처리하는 것이 효과적이다.
* 이는 directed graph를 undirected graph로 바꿈으로써 수행할 수 있다.
* Directed graph의 결합 확률 분포는, 각 clique의 분포가 $p(x_v \vert x_{\pi_v})$의 곱으로 나타난 undirected graph 결합 확률 분포의 형태로 볼 수있다.
* 그런데 여기서 문제는 어떤 node $i$에 대한 parent $\pi_i$들은 서로 연결되어있지 않은 경우가 많고, 이 경우 같은 clique에 속하는 것으로 보기 힘들다는 점이다.
  * 이는 parent $\pi_i$들 사이에 node가 없는 곳을 (undirected) edge로 채워줌으로써 해결한다.
  * 이렇게 하면 $p(x_i \vert x_{\pi_i})$의 모든 argument들이 한 clique에 소속되게 됨. 이를 *moral graph*라고 부른다.
* 또한 계산적인 측면에서 보면 조건부 확률분포를 찾는 것은 큰 문제가 아니다.
  * 만약 node들의 부분집합 $E$에 대해  $\{ X_E = x_E\}$의 사건 하에서의 조건부 확률분포를 보고 싶다면, 
  * ????????????????

* Marginalization을 수행하는 것의 계산 복잡도를 줄이는 방법을 찾는 것이 주 목표.
  * 이에는 크게 exact algorithms, sampling algorithms, variational algorithms의 접근법이 있다.
  
<br>


## 1.3.1 Exact algorithms

## 1.3.1.1 Elimination algorithm

* Marginalization을 수행하면 그래프는 어떻게 변화할까?
* 한 변수를 marginalize out하면 marginalize out된 변수와 인접한 모든 변수들은 clique가 되는 효과가 있다.
  * Node 2,3,4가 node 5와 인접한 다음 상황에서 node 5를 marginalize하면, 아래와 같이 Node 2,3,4가 하나의 clique가 되는 효과.

![image-20200319141214804](https://user-images.githubusercontent.com/45325895/77082439-664c9400-6a3f-11ea-99b8-da018fe1f4c0.png){: .center-image}

$$
\begin{align*}
\sum_{x_5}\psi(x_1,x_2)\psi(x_1,x_3)\psi(x_2,x_5)\psi(x_3,x_4, x_5) &= \psi(x_1,x_2)\psi(x_1,x_3)\left(\sum_{x_5}\psi(x_2,x_5)\psi(x_3,x_4, x_5) \right) \\
&= \psi(x_1,x_2)\psi(x_1,x_3)m_5(x_2,x_3,x_4) 

\end{align*}
$$

* 따라서 marginalization을 수행하면 새로 등장하는 edge들이 생긴다.
* 이 새로 등장한 edge들을 $\mathcal E^\prime$이라고 하면, 원래 graph $(\mathcal V, \mathcal E )$에 대해, $(\mathcal V, \mathcal E \cup \mathcal E^\prime)$을 *reconstituted graph*라고 한다.
  * 이 작업을 *triangulation*이라고 한다.
  * 어떤 순서로 marginalization을 수행하는지에 따라 그에 대한 reconstituted graph도 달라짐.
* 다음과 같은 사실이 알려져 있다.
  * Reconstituted graph의 largest clique의 size가 계산복잡도를 결정.
    * 이 maximal clique의 size $- 1$를 그 graph의 *treewidth*라고 부른다.
  * Reconstituted graph는 chordal graph.
* Marginalization을 수행할 때 변수들을 marginalize out하는 순서를 *elimination order*라고 한다.
* Largest clique의 size를 최소화하는 최적의 elimination order를 찾는 문제는 NP-hard임이 알려져 있다.
* Optimal 혹은 good enough한 elimination order를 찾는 방법으로는 *probabilistic elimination* 등이 있다.
* 이러한 elimination approach의 단점은 이렇게 해서 얻을 수 있는 것은 하나의 marginal probability라는 점.
  * 여러 marginal probability를 얻으려면 이 전체 작업을 여러번 수행해야 한다.



## 1.3.1.2 Approaches available under tree structure

* Undirected tree 구조에서 clique는 두 node의 쌍, 혹은 다른 어느 node와도 연결되지 않은 singleton이다.
* Tree의 joint probability distribution은 potential $\{ \psi(x_i, x_j) : (i,j) \in \mathcal E\}$와 $\{ \psi(x_i) : (i) \in \mathcal V\}$로 parameterize된다.

## 1.3.1.2.1 Sum-product algorithm

* Children node보다 parent node가 먼저 marginalize out되지 않는 elimination order를 생각해보자.

![image-20200319141108602](https://user-images.githubusercontent.com/45325895/77082539-8f6d2480-6a3f-11ea-9ce8-b5ab5b67f911.png){: .center-image}

* 위와 같이 node $j$와 node $i$가 child-parent 관계이고, 이 때 node $j$를 marginalize out하여 전달되는 *message*를 다음과 같이 recursive하게 나타낸다.

$$
\begin{align*}
m_{ji}(x_i) &= \sum_{x_j}\Bigg( \psi(x_j)\psi(x_i,x_j) m_{kj}(x_j)  m_{lj}(x_j) \Bigg) \\
&= \sum_{x_j}\left( \psi(x_j)\psi(x_i,x_j) \prod_{k \in \mathcal N(j) \backslash i} m_{kj}(x_j) \right)
\end{align*}
$$

* 이 때, $\mathcal N(j)$는 node $j$의 neighborhood인 node들의 집합을 나타낸다.
  * $\mathcal N(j) \backslash i$는 node $j$의 child인 node들의 집합을 나타낸다.

* Tree의 root node $f$에 대한 marginal probability는 다음과 같다.

$$
p(x_f) \propto \psi(x_f) \prod_{e \in \mathcal N(f)} m_{ef}(x_f)
$$

* Root가 아닌 다른 node $i$의 marginal probability는 어떻게 구할까?
* 다음 두 사실에 주목하자.
  * 위에서 구한 message의 식이 방향성을 갖는다
  * Undirected tree는 tree의 임의의 node를 root로 볼 수 있다.
* Edge의 set $\mathcal E$에 대해, 모든 message를 다 구하면 총 $2\vert \mathcal E\vert$개가 된다.
* 이 message를 다 구하면 위의 root node에 대한 marginal probability 공식에 의해, tree 내의 모든 node에 대한 marginal probability를 구할 수 있다.
* 이를 *sum-product algorithm*이라고 한다.
  * 우리의 graphical model이 tree일 때 general marginal probability를 구하는 방법을 제공한다.



## 1.3.1.2.2 Junction tree algorithm

* Elimination algorithm과 sum-product algorithm을 합친 것으로 볼 수 있다.
* Clique들을 component로 본 *hypergraph*를 tree 구조로 보고 sum-product algorithm을 수행한다.
  * Single node들 간의 message가 아니라 clique 간의 message를 계산하게 됨.
* 이 때 clique는 원래 graph의 clique이 아니라, 특정 elimination order에 의해 생성된 reconstituted graph의 clique이다.

<br>

## 1.3.2 Sampling algorithms

* 1.3.1의 방법은 graph theory 혹은 graph의 특정 structure를 이용하여 graphical model의 probabilistic inference를 수행함.
* *Importance sampling*, *Markov chain Monte Carlo (MCMC)*와 같은 방법을 이용할 수 있다.
* MCMC의 대표적인 모형인 *Gibbs sampling*의 경우에는 각 individual variable의 conditional이 사용 가능해야 한다.
  * 여기서 *conditional*이란 다른 모든 변수에 대한 한 변수의 조건부 확률 분포이다.
  * $X_1, \ldots, X_k$를 변수로 갖는 모형을 추정한다고 하면, $X_1$의 conditional은 $X_1 \vert X_2, \ldots , X_k$를 의미한다. 
* Graphical model의 Markov property는 이에 매우 적절하다.
  * Directed graphical model에서 어떤 node의 Markov blanket은 그 node의 parent node들의 집합이다.
  * Undirected graphical model에서 어떤 node의 Markov blanket은 그 node의 neighborhood들의 집합이다.

<br>

## 1.3.3 Variational algorithms

* *Variational inference*는 어떤 목표 확률 분포에 대한 approximation을 최적화 문제로 바꾸어 해결한다.
* 목표 확률 분포의 parameter가 아닌, variational parameter로 parameterized된 더 간단한 확률 분포의 class에서 최적화를 수행.
  * 많은 경우, 이 확률 분포의 class는 실제 목표 확률 분포를 포함하지 않는다.
* Variational distribution과 목표 확률 분포의 *Kullback-Leibler (KL) divergence*를 최소화하는 variational parameter를 찾게 된다.
* 이에 더해 "확률 분포"들의 class에서 최적화를 수행한다는 제약을 해제하여, 목표 확률 분포에 대한 더 좋은 성능의 근사를 도모하는 방법이 제시되었다.
  * Wainwright & Jordan, *"Graphical Models, Exponential Families, and Variational Inference"*, 2008.

  

* 그 방법을 간단히 소개하면 다음과 같다.
* Exponential family에 속하는 유한개의 모수를 갖는 확률 분포를 생각해보자.
* 만약 위에서 소개한 directed case, undirected case의 결합 확률 분포의 factor가 exponential family에 속한다고 가정하면, 결합 확률 분포를 다음과 같이 나타낼 수 있다.

$$
p(x_\mathcal V \vert \theta) = \exp\Big\{ \langle \theta, \phi(x_\mathcal V)\rangle - A(\theta) \Big\} \\
A(\theta) = \log \int \exp\big\{ \langle \theta, \phi(x_\mathcal V)\rangle \big\} \nu(dx_\mathcal V)
$$

* 이 경우 $A(\theta)$를 approximate함으로써 우리는 $p(x_\mathcal V)$를 approximate할 수 있다.
* 먼저 다음 두 사실을 이용한다.
  * Parameter space $\Theta$가 convex라면, cumulant generating function $A(\theta)$는 convex function이다.
  * 임의의 convex function은 *conjugate dual function*에 대해 variational form으로 나타낼 수 있다.
* $A^\ast(\mu)$는 $A(\theta )$의 *(Fenchel) conjugate dual function*을 의미한다.

$$
A^\ast (\mu) = \sup_{\theta \in \Theta} \Big\{ \langle \theta, \mu \rangle - A(\theta)\Big\}
$$

![image-20200319170839933](https://user-images.githubusercontent.com/45325895/77082624-af9ce380-6a3f-11ea-8698-7ba5daef6bbe.png){: .center-image}

* Fenchel conjugate dual function은 다음과 같이 이해할 수 있다.
  * $x^\ast =-4$에 대한 $f^\ast(x^\ast)$의 값은, original 함수 $f(x)$와 같은 공간에 원점을 지나고 기울기가 $x^\ast = -4$인 직선을 그렸을 때, 그 직선을 함수값 축 방향으로 얼마나 평행 이동해야 $f(x)$와 접하게 되는지를 나타낸다.
  * 어떤 $\mu$에 대한 $A^\ast(\mu)$의 값은, original 함수 $A(\theta)$와 같은 공간에 기울기 벡터가 $\mu$인 linear function을 그렸을 때, 그 linear function을 함수값 축 방향으로 얼마나 평행 이동해야 $A(\theta)$와 접하게 되는지를 나타낸다.

* 만약 어떤 함수 $f$가 closed, proper convex function이라면, $f^{\ast\ast} = f$가 만족함이 알려져 있다.

  * $f$ is a *proper convex function* if $f$ is convex, $f(x)< +\infty$ for at least one $x$, and $f(x)>-\infty$ for every $x$.
  * $f$ is a *closed function* if the following set $A_\alpha$ is closed for each $\alpha \in \mathbb R$.

$$
A_\alpha = \{ x \in \text{dom} f \vert f(x) \leq \alpha\}
$$

* 이를 이용하면 다음과 같은 형태로 cumulant generating function $A(\theta)$를 나타낼 수 있다.

  * 이 때, $\mathcal M$은 realizable mean parameter의 집합을 의미한다.

$$
A(\theta) = \sup_{\mu \in \mathcal M} \Big\{ \langle \theta, \mu \rangle - A^\ast (\mu)\Big\} \\
\mathcal M = \left\{ \mu \in \mathbb R^d \enspace \Big\vert \enspace \exists p(\cdot) \enspace \text{ s.t. } \enspace \mathbb E_p[ \phi(X_\mathcal V) ]= \mu  \right\}
$$

* 이제 우리는 확률분포의 특정 class로 최적화 수행 범위를 제한하고, 그 class 내에서 가장 목표 확률 분포와 가장 가까운 variational distribution을 찾는다.

  * Variational distribution의 class를 특정했다면, sufficient statistic $\phi(X_\mathcal V)$의 기대값을 구함으로써 그에 따른 mean parameter의 집합 $\mathcal M_{\text{Tract}}$를 구할 수 있다.
  * 이 $\mathcal M_{\text{Tract}}$ 내에서 다음 optimization 문제를 해결한다.

$$
\sup_{\mu \in \mathcal M_{\text{Tract}}} \Big\{ \langle \theta, \mu \rangle - A^\ast (\mu)\Big\}
$$

  * 이 optimization의 해 $\mu^\ast$는 expected sufficient statistic의 approximation이 된다.

* 이와 같이 $A(\theta)$를 approximate, 즉 $p(x_\mathcal V)$를 approximate할 수 있다.

<br>
<br>

다음 section들은 graphical model의 다양한 application과 대표적인 연구들을 소개한다. 여기서는 생략.

# 1.4 Bioinformatics

# 1.5 Error-control codes

# 1.6 Speech, language and information retrieval

## 1.6.1 Markov and hidden Markov models

## 1.6.2 Variations on Markovian models

## 1.6.3 A hierarchical Bayesian model for document collection

