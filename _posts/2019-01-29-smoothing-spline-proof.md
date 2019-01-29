---
layout: post
title: "Heuristic Derivation of Smoothing Spline"
tags: [Statistics]
comments: true
---

이 포스트는 Smoothing spline에 대한 기본적인 이해를 갖고 있는 독자를 대상으로 한다. Piecewise polynomial, natural cubic spline, smoothing spline 등의 개념이 익숙치 않다면 The Elements of Statistical Learning의 Chapter 5.1 ~ 5.4, 혹은 이를 정리한 나의 포스트(**[Link](https://lee-jaejoon.github.io/ESL-5/)**)를 참고하면 도움이 될 것이다.  
<br>
# Regression Problem with Curvature Constraint

Observed data, $(x_1,y_1),(x_2,y_2),...,(x_N,y_N)$를 통해 regression function을 추정하기 위하여, 아래와 같은 식을 최소화하는 $\hat{f}$를 찾는 문제를 생각해보자. $f$는 $[a,b]$에서 정의되고, $f''(t)^2 $가 적분가능한 임의의 함수이다.

$$
\hat{f}=\text{arg}\min_{f} \enspace \sum_{i=1}^{N} { \{ y_i-f(x_i) \} }^2 + \lambda \int_a^b { \{ f''(t) \} }^2 dt
$$

 * 첫 번째 항 $\sum_{i=1}^{N} { \{ y_i-f(x_i) \} }^2$은 **fit이 observed data와 얼마나 가까운지**를 측정하는 기준이 된다.

 * 두 번째 항은 함수 $f$의 **curvature**, 즉 $f$의 굴곡진 정도를 제한하고 spline을 smoothing하는 기준이 된다.
	 * $f(x)=a+bx$와 같이 $f$의 curvature가 전혀 없을 때, 즉 이계도함수 $f'' $이 $y=0$일 때, 아래와 같이 두번째 항은 0이 된다.

 * $\lambda$는 위 식 전체를 최소화함에 있어, 두 기준들 간의 비중을 정해주는 smoothing parameter이다.
	 * $\lambda=0$일 때, $f$는 전혀 smoothing되지 않고 모든 observed points $(x_1,y_1),...,(x_N,y_N)$를 지나가는 곡선이 될 것.
	 * $\lambda=\infty$일 때, $f$는 이계도함수가 $0$이 되어야하므로 least square로 직선(일차함수)을 fit한 결과가 나올 것.

따라서, 주어진 $\lambda$에 대해서, 위 식을 최소화하는 함수 $\hat{f}$는 observed data와의 fit도 좋으면서, curvature도 크지 않은 **smooth**한 함수가 될 것이다. 그런데 위의 문제에서는 $f''(t)^2 $가 적분가능하다는 것 외에는 $f$에 대해 어떠한 제한도 두지 않았다. **$f''(t)^2 $가 적분가능한 함수는 무수히 많을 것이고, 이를 모두 고려하는 것은 불가능하다.** 우리는 위 식을 최소화하는 함수, $\hat{f}$를 어떻게 찾아야 할까?  
<br>
<br>

# Heuristic Derivation

다행히도 위 최소화 문제는 **$x_1,x_2, ...,x_N$에서 knot을 갖는 natural cubic spline**을 **유일한 해**로 갖는다는 것이 밝혀져 있다 (증명 **[Reinsch (1967)](https://link.springer.com/content/pdf/10.1007/BF02162161.pdf)**). 따라서 우리는 $x_1,x_2, ...,x_N$에서 knot을 가지면서 위 식을 최소화하는 Natural cubic spline을 **Smoothing spline**이라고 정의한다. 하지만 training data의 input point들에서 knot을 갖는 natural cubic spline이 항상 위 최소화 문제의 해가 된다는 것은 직관적으로도 쉽게 다가오지 않는다. 또한, 엄밀한 증명(**[Reinsch (1967)](https://link.springer.com/content/pdf/10.1007/BF02162161.pdf)**)을 읽고 이해하는 것도 쉽지 않은 일이다. 그래서 이 포스트에서는 아래와 같은 Theorem을 증명함으로써 이에 대한 Heuristic derivation을 소개하고자 한다.  

### Theorem
$[a,b]$에서 정의되고 $g''^2$가 적분가능한 한 함수 $g$를 생각해보자. 그리고 $\tilde{g}$를 $[a,b]$ 내의 $N$개의 data points $x_1, x_2, ..., x_N$에서 knot을 갖고, 아래 식을 만족하는 natural cubic spline이라고 하자. ($a < x_1 < x_2 < ...< x_N < b$)

$$
g(x_i)=\tilde{g}(x_i) \enspace , \enspace \enspace \text{for} \enspace  i=1,2,...,N
$$

그렇다면 다음이 만족한다.

$$
\int_a^b g''^2(t)dt \ge \int_a^b \tilde{g}''^2 (t)dt
$$

### Meaning

임의의 함수에 대해 정의된 regression problem의 해가 항상 natural cubic spline이라는 것은 위 theorem으로 보일 수 있다. Theorem에서 나온 적분식이 위 regression problem의 curvature 제약식과 정확히 일치한다는 점을 확인하자. 이 Theorem을 적용하는 논리는 아래와 같다.  
  
 1. 위 regression problem의 첫 번째 항, $\sum_{i=1}^{N} { \{ y_i-f(x_i) \} }^2$를 최소화하는 어떤 함수 $g$가 있다고 하자.
 2. 그렇다면 그 $g$에 대하여, curvature 제약 항의 값이 더 작은 natural cubic spline, $\tilde{g}$가 항상 존재한다.
	 * $\tilde{g}(x_i)=g(x_i)\enspace , \enspace \enspace \text{for} \enspace  i=1,2,...,N$를 만족하는 natural cubic spline은 언제나 만들 수 있기 때문이다.
 3. 따라서, 위 regression problem의 해는 natural cubic spline의 형태를 갖는다.  
  
이 theorem을 이용한 위 설명이 엄밀한 증명이 아닌 Heuristic Derivation인 이유는, $\sum_{i=1}^{N} { \{ y_i-f(x_i) \} }^2$를 최소화하는 함수 $g$가 존재한다는 1번 statement를 증명하지 않았고, Natural cubic spline의 형태로 존재하는 해의 유일성에 대해서도 증명을 하지 않았기 때문이다.  

<br>

### Proof of the Theorem
임의의 $x \in [a,b]$에 대해, $h(x)=g(x)-\tilde{g}(x)$로 정의된 함수를 $h$라고 하자. $g(x)=h(x)+\tilde{g}(x)$이므로,
$$
\enspace \enspace \enspace \enspace \int_a^b g''^2(t)dt = \int_a^b h''(t)^2+2 h''(t) \tilde{g}''(t)+\tilde{g}''^2(t) dt \enspace \enspace \cdots \cdots(a)
$$

여기서 부분적분을 사용하면 아래와 같이 정리할 수 있다.

$$
\int_a^b h''(t) \tilde{g}''(t) dt = \left. h'(t) \tilde{g}''(t) \right|_{a}^{b} - \int_a^b h'(t) \tilde{g}'''(t) dt
$$

$$
= h'(b) \tilde{g}''(b) -h'(a) \tilde{g}''(a) - \int_a^b h'(t) \tilde{g}'''(t) dt
$$

natural cubic spline은 boundary knot (가장 바깥쪽에 있는 knot. 즉, $x_1,x_N$)의 바깥에서 linear한 cubic spline으로 정의된다. 즉,  $[a,x_1]$과 $[x_N,b]$에서 $g(x)$는 일차함수 꼴이며, 두번 미분하면 0이 된다. 따라서, $\tilde{g}''(a)=\tilde{g}''(b)=0$이므로,

$$
\int_a^b h''(t) \tilde{g}''(t) dt = - \int_a^b h'(t) \tilde{g}'''(t) dt
$$

$$
= - \int_a^{x_1} h'(t) \tilde{g}'''(t) dt - \int_{x_1}^{x_2} h'(t) \tilde{g}'''(t) dt - \cdots - \int_{x_N}^{b} h'(t) \tilde{g}'''(t) dt
$$

Natural cubic spline은 knot과 knot 사이에서는 3차 이하의 다항식으로 정의되는 piecewise polynomial이다. 또한, 3차 이하의 다항식은 세 번 미분했을 때 상수가 된다. 따라서 위와 같이 적분 구간을 knot 단위로 나누어 준 뒤에는 아래와 같이 쓸 수 있다.

$$
= - \tilde{g}'''(a+)\int_a^{x_1} h'(t)  dt - \tilde{g}'''(x_1 +) \int_{x_1}^{x_2} h'(t) dt - \cdots - \tilde{g}'''(x_N +)\int_{x_N}^{b} h'(t) dt
$$

위에서 언급한 것과 같이, $[a,x_1]$과 $[x_N,b]$에서 $g(x)$는 일차함수 꼴이며, 세 번 미분하면 0이 된다. $ \tilde{g}'''(a+)= \tilde{g}'''(x_N +)=0$. 따라서,
$$
= c_1 \int_{x_1}^{x_2} h'(t) dt - \cdots +c_{N-1} \int_{x_{N-1}}^{x_N} h'(t) dt
$$

$$
= c_0[h(x_2)-h(x_1)]+ \cdots +c_N [h(x_N)-h(x_{N-1})]
$$

$g(x_i)=\tilde{g}(x_i)$ for all $i$이므로, $h(x_i)=g(x_i)-\tilde{g}(x_i)=0$ for all $i$이다. 따라서,

$$
\int_a^b h''(t) \tilde{g}''(t) dt = 0
$$

이를 위의 식 $(a)$에 대입하면 다음을 얻을 수 있다.

$$
\int_a^b g''^2(t)dt = \int_a^b h''(t)^2 dt + \int_a^b  \tilde{g}''^2(t) dt
$$

$$
\int_a^b g''^2(t)dt \ge \int_a^b  \tilde{g}''^2(t) dt
$$

<br>
<br>

# Reference
> Hastie, T., Tibshirani, R.,, Friedman, J. (2001). The Elements of Statistical Learning. New York, NY, USA: Springer New York Inc.. 
> http://publish.illinois.edu/liangf/files/2016/05/Note_splines.pdf

