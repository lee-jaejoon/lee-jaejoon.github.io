---
layout: post
title: "Lagrangian Dual Problem & Karush-Kuhn-Tucker Conditions"
tags: [Optimization]
comments: true
---

통계학 혹은 머신러닝의 대부분의 모형에서 모형의 학습은, 어떤 목적함수를 최소화(혹은 최대화)하여 모형의 parameter의 최적 값을 찾음으로써 이루어진다. Lagrangian method는 제약 하 최적화 문제를 가장 대표적인 방법 중 하나이다. 이 포스트에서는 Lagrangian dual problem에 대한 이해, 그리고 그 과정에서 필요한 최적화 이론의 몇 가지 개념들을 함께 소개하고자 한다. 또한 포스트의 후반부에서는 Karush-Kuhn-Tucker 조건에 대해 소개한다. strong duality가 보장이 되었을 때, Karush-Kuhn-Tucker 조건은 어떤 벡터들이 최적화 문제의 primal solution과 dual solution이 되는 것과 필요충분 관계의 조건이 된다는 점에서 매우 의미가 있는 조건이다.

# 1. Lagrangian Dual Problem

다음과 같은 최적화 문제를 해결하는 것이 목표라고 하자. "원래" 해결하고자 했던 문제라는 뜻에서 이를 *Primal problem*이라고 부른다.

$$
\begin{equation*}
\begin{aligned}
& \underset{\mathbf{x}}{\text{minimize}}
& &  f(\mathbf{x}) \\
& \text{subject to}
& & h_i(\mathbf{x}) \le 0 , \enspace i=1,\cdots,m \\
&&& \ell_j(\mathbf{x}) = 0 , \enspace j=1,\cdots,r.
\end{aligned}
\end{equation*}
$$

이 최적화 문제는 convex optimization problem이 아니어도 된다. 다만 만약 primal problem이 convex problem인 경우 Lagrangian dual problem이 어떻게 되는지는 뒤에서 한 번 더 다뤄보겠다. 위 primal problem으로부터, 아래와 같은 함수 $L(\mathbf{x}, \mathbf{u}, \mathbf{v})$를 정의하자. 

$$
L(\mathbf{x}, \mathbf{u}, \mathbf{v})=f(\mathbf{x})+\sum_{i=1}^{m} u_i h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{x})
$$

$$
\text{where } \enspace \mathbf{u} \in \mathbb{R}^m, \mathbf{v}\in \mathbb{R}^r,u_i>0 \text{ for all } i
$$

이 함수 $L(\mathbf{x}, \mathbf{u}, \mathbf{v})$는 다음과 같은 중요한 성질이 있다.

$$
f(\mathbf{x}) \ge L(\mathbf{x},\mathbf{u}, \mathbf{v}) \enspace \text{ at each feasible } \mathbf{x}
$$

여기서 최적화 목적함수 $f$의 어떤 input point $\mathbf{x}$가 *feasible*(실현 가능한)하다는 것은 "제약 조건을 만족하는", 다시 말해서 "최적화 문제의 해가 될 수 있는" $\mathbf{x}$라는 의미이다. 즉, primal problem의 제약 조건을 만족하는 모든 $\mathbf{x}$에 대해 위 식이 성립한다는 것이다. 위 식이 성립하는 이유는 아래와 같다. Feasible한 $\mathbf{x}$는 다음을 만족한다.

$$
h_i(\mathbf{x}) \le 0 , \enspace i=1,\cdots,m, \enspace \enspace \ell_j(\mathbf{x}) = 0 , \enspace j=1,\cdots,r.
$$

이를 $L(\mathbf{x}, \mathbf{u}, \mathbf{v})$의 식에 대입하면, 위에서 말한 부등식을 얻을 수 있다. 사실 위에서 $L(\mathbf{x}, \mathbf{u}, \mathbf{v})$를 정의할 때, $\mathbf{u}$가 모든 $i$에 대해 $u_i>0$를 만족하도록 한 이유가 바로 아래의 부등식을 만족하게 하기 위해서이다.

$$
L(\mathbf{x}, \mathbf{u}, \mathbf{v})=f(\mathbf{x})+\sum_{i=1}^{m} u_i h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{x}) \le f(\mathbf{x})  \enspace \text{ at each feasible } \mathbf{x}
$$

Feasible한 $\mathbf{x}$를 모두 모은 집합을 $C$라고 하자. 그리고 primal problem의 optimal value를 $f^\ast$라고 하자.

$$
f^\ast = \min_{\mathbf{x} \in C} f(\mathbf{x})
$$

다음과 같이 $L(\mathbf{x}, \mathbf{u}, \mathbf{v})$를 $\mathbf{x}$에 대해 최소화하면, $f$의 optimal value $f^\ast$에 대한 다음과 같은 부등식을 얻을 수 있다. 또한, 우리는 이를 통해 $u_i>0 \text{ for all } i$를 만족하는 $\mathbf{u}, \mathbf{v}$에 대해 정의된 함수, $g(\mathbf{u}, \mathbf{v})$를 정의한다.


$$
f^\ast = \min_{\mathbf{x} \in C} f(\mathbf{x}) \ge  \min_{\mathbf{x} \in C} L(\mathbf{x}, \mathbf{u}, \mathbf{v}) \ge \min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}, \mathbf{v}) := g(\mathbf{u}, \mathbf{v})
$$

이와 같이 정의된 함수 $g(\mathbf{u}, \mathbf{v})$를 *Lagrange dual function*이라고 부른다.

$$
g(\mathbf{u}, \mathbf{v})=\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}, \mathbf{v})
$$

$$
\text{where } \enspace \mathbf{u} \in \mathbb{R}^m, \mathbf{v}\in \mathbb{R}^r,u_i>0 \text{ for all } i
$$

Lagrange dual function, $g(\mathbf{u}, \mathbf{v})$는 다음을 만족한다.

$$
f^\ast \ge g(\mathbf{u}, \mathbf{v}) \enspace \text{ for all } \mathbf{u}, \mathbf{v} \text{ such that } u_i>0, \forall  i
$$

다시 말해서, **$g(\mathbf{u}, \mathbf{v})$는 $f$의 optimal value, $f^\ast$에 대한 lower bound가 된다.** 또한, 이 Lagrange dual function에 대해 다음과 같은 최적화 문제를 생각해보자.

$$
\begin{equation*}
\begin{aligned}
& \underset{\mathbf{u}, \mathbf{v}}{\text{maximize}}
& &  g(\mathbf{u},\mathbf{v}) \\
& \text{subject to}
& & u_i>0, \enspace i=1,\cdots,m.
\end{aligned}
\end{equation*}
$$

Primal problem과 짝을 이루어 존재하는 이와 같은 최적화 문제를 *Lagrange dual problem*이라고 부른다. Lagrange dual problem은 다음과 같은 몇 가지 중요한 성질이 있다.

 * **Weak duality** : 만약 Lagrange dual problem의 optimal value를 $g^\ast=\max_{\mathbf{u}, \mathbf{v}} g(\mathbf{u},\mathbf{v})$라고 한다면, 다음이 항상 성립한다.
	 * 이는 primal problem이 convex problem인지 여부와는 상관없이, 임의의 최적화 문제에 대해 항상 성립한다.

$$
f^\ast \ge g^\ast
$$

$$
\because \enspace \enspace f^\ast \ge g(\mathbf{u}, \mathbf{v}) \enspace \text{ for all } \mathbf{u}, \mathbf{v} \text{ such that } u_i>0, \forall  i
$$

 * Lagrange dual problem은 convex optimization problem이다. 
	 * 이 또한, primal problem이 convex problem인지 여부와는 상관없이, 임의의 최적화 문제에 대해 항상 성립하는 성질이다.

원래 우리의 목표는 $f(\mathbf{x})$의 최소값, $f^\ast$를 찾는 것이었고, 이를 최적화 문제로 나타낸 것이 primal problem이었다. 그와 짝을 이루어 존재하는 Lagrange dual problem은, $f(\mathbf{x})$를 직접 최소화하는 것이 아니라, $f^\ast$의 하한, $g(\mathbf{u},\mathbf{v})$를 최대화하는 방법으로 $f^\ast$를 찾고자 하는 것이다. 그런데 위에서 확인한 weak duality 성질은 임의의 최적화 문제의 dual optimal value가 primal optimal value보다 작거나 같다는 것만 보장할 뿐, 실제로 dual optimal value가 primal optimal value와 같아질 수 있는지 여부는 말해주지 않는다. 

## Strong Duality

우리는 Lagrange dual problem의 optimal value $g^\ast$가 항상 primal problem의 optimal value $f^\ast$보다 작거나 같다는 것을 확인했고, 이를 Weak duality라고 했다. Weak duality는 임의의 최적화 문제에 대해 항상 성립하는 것이다. 이에 더해 특정 조건이 만족된다면, 어떤 최적화 문제에서는 다음이 성립한다.

$$
f^\ast = g^\ast 
$$

이를 **strong duality**라고 한다. Lagrange dual problem을 구하는 것은, $f^\ast$의 하한을 최대화 하는 것과 같은데, strong duality는 하한을 최대한 밀어올렸을 때의 값이 primal problem의 optimal value $f^\ast$와 같아짐을 의미한다.  
  
이와 같은 **strong duality**가 성립하도록 하는 충분조건안 **Slater's condition**을 소개하고자 한다. 이는 충분조건이므로, 어떤 최적화 문제가 Slater's condition을 만족한다면 이 문제는 strong duality가 성립하지만, strong duality가 성립하는 모든 최적화 문제가 Slater's condition을 만족하는 것은 아니다.

 * **Slater's condition** : 만약 primal problem이 convex optimization problem이고 $( \text{i.e., }f, h_1, \cdots, h_m$이 convex function, $\ell_1, \cdots, \ell_r$이 affine function$)$, strictly feasible한 $( \text{i.e., }h_1(\mathbf{x})<0,\cdots,h_m(\mathbf{x})<0, \ell_1(\mathbf{x})=0,\cdots,\ell_r(\mathbf{x})=0)$ input point $\mathbf{x}$가 존재한다면, strong duality가 만족한다.

 * Slater's condition은 일부 완화가 가능한데, strict feasibility 조건 중 부등식제약에 대한 strict feasibility 조건은 affine이 아닌 $h_j$에 대해서만 만족해도 된다. 다시 말해서, Affine인 부등식제약 $h_j$는 strict feasibility를 만족하지 않아도 괜찮다.  
<br>
<br>
<br>




# 2. Karush-Kuhn-Tucker Conditions

우리가 해결하고자 하는 최적화 문제, primal problem이 아래와 같다고 하자.

$$
\begin{equation*}
\begin{aligned}
& \underset{\mathbf{x}}{\text{minimize}}
& &  f(\mathbf{x}) \\
& \text{subject to}
& & h_i(\mathbf{x}) \le 0 , \enspace i=1,\cdots,m \\
&&& \ell_j(\mathbf{x}) = 0 , \enspace j=1,\cdots,r.
\end{aligned}
\end{equation*}
$$

이에 대해, 다음과 같은 함수를 정의할 수 있다.

$$
L(\mathbf{x}, \mathbf{u}, \mathbf{v})=f(\mathbf{x})+\sum_{i=1}^{m} u_i h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{x})
$$

$$
\text{where } \enspace \mathbf{u} \in \mathbb{R}^m, \mathbf{v}\in \mathbb{R}^r,u_i>0 \text{ for all } i
$$

$g(\mathbf{u}, \mathbf{v})=\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}, \mathbf{v})$로 정의된 함수 $g(\mathbf{u}, \mathbf{v})$에 대한 Lagrange dual problem은 아래와 같다.

$$
\begin{equation*}
\begin{aligned}
& \underset{\mathbf{u}, \mathbf{v}}{\text{maximize}}
& &  g(\mathbf{u},\mathbf{v}) \\
& \text{subject to}
& & u_i>0, \enspace i=1,\cdots,m.
\end{aligned}
\end{equation*}
$$


Primal problem과 dual problem이 위에서와 같을 때, Karush-Kuhn-Tucker 조건(이하 KKT 조건)은 다음 네 조건을 말한다.

$\text{1. (stationarity) } \enspace 0 \in \partial \Big( f(\mathbf{x})+\sum_{i=1}^{m} u_i h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{x}) \Big)$  
$\text{2. (complementary slackness) } \enspace u_i \cdot h_i(\mathbf{x})=0 \enspace \text{ for all } i$  
$\text{3. (primal feasibility) } \enspace h_i(\mathbf{x}) \le 0, \enspace \ell_j(\mathbf{x}) = 0 \enspace \text{ for all } i,j $  
$\text{4. (dual feasibility) } \enspace u_i \ge 0 \enspace \text{ for all } i$  
  
우리의 목표는 Strong duality가 성립하는 최적화 문제에 대해 다음이 성립하는 것을 보이는 것이다.

$$
\mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ are primal and dual solutions}
$$

$$
\iff \mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ satisfy KKT conditions}
$$

Slater's condition이나 다른 regularity condition을 통해 우리의 최적화 문제가 Strong duality가 성립한다는 사실을 알 수 있다고 하자. 이 때, 최적화 문제의 해, $\mathbf{x}^\ast$가 KKT 조건을 만족하는 것이 보장이 되는 것이다. 또한 반대로, $\mathbf{x}^\ast$가 KKT 조건을 만족한다면, 이는 최적화 문제의 해가 된다는 것을 알 수 있다. 보다 자세한 설명에 들어가기에 앞서, 짚고 넘어가야 할 수학적 개념이 있다. 1번 stationarity 조건의 $\partial$ 기호로 나타내어진 subdifferential, 그리고 subgradient에 대해 간단히 알아보자.

# Digression: Subgradient & Subdifferential

Subgradient란 gradient의 개념을 non-smooth 함수에까지 확장한 개념이다. 함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$에 대해 다음을 만족하는 벡터 $g \in \mathbb{R}^n$를 $\mathbf{x}$에서의 $f$의 *subgradient*라고 한다.

$$
f(\mathbf{z}) \ge f(\mathbf{x}) + g^T (\mathbf{z}-\mathbf{x}) \enspace \text{ for all }\mathbf{z} \in \mathbb{R}^n
$$

또한, $\mathbf{x} \in \mathbb{R}^n$에서의 subgradient를 모두 모은 집합을 $\mathbf{x}$에서의 $f$의 *subdifferential*, $\partial f(\mathbf{x})$라고 한다.

$$
\partial f(\mathbf{x})= \{g \enspace | \enspace g \text{ is a subgradient of } f \text{ at } \mathbf{x} \}
$$

<br>
### Derivative as the Best Affine Approximation

함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$의 $\mathbf{x} \in \mathbb{R}^n$에서의 미분계수(derivative), $Df(x) \in \mathbb R^{1 \times n}$는 다음을 만족하는 $n$차원 행벡터로 정의된다.

$$
\frac{||f(\mathbf{z}) - f(\mathbf{x})- Df(\mathbf{x}) (\mathbf{z}-\mathbf{x})||}{||\mathbf{z}-\mathbf{x}||} \rightarrow 0 , \text{ as } ||\mathbf{z}-\mathbf{x}|| \rightarrow 0
$$

$$
Df(\mathbf{x})= \Big[ \left. \frac{\partial f(\mathbf{z})}{\partial z_1} \right|_{\mathbf{z}=\mathbf{x}} \cdots  \left. \frac{\partial f(\mathbf{z})}{\partial z_n} \right|_{\mathbf{z}=\mathbf{x}} \Big]
$$

이 때, 다음이 성립한다.

$$
||f(\mathbf{z}) - f(\mathbf{x})- Df(x) (\mathbf{z}-\mathbf{x})|| \rightarrow 0 , \text{ as } ||\mathbf{z}-\mathbf{x}|| \rightarrow 0
$$

위 성질에 의해, $\mathbf{x}$의 근방에서는 아래와 같은 식이 성립한다. 여기서 $\nabla f(\mathbf{x})$는 $f$의 $\mathbf{x}$에서의 gradient로, $\nabla f(\mathbf{x})=Df(x)^T$로 정의된다.

$$
f(\mathbf{z}) \approx f(\mathbf{x})+ \nabla f(\mathbf{x})^T (\mathbf{z}-\mathbf{x})
$$

따라서, $\mathbf{x}\in \mathbb{R}^n$에서의 gradient, 혹은 $\mathbf{x}$에서의 미분계수(derivative)는, 함수 $f$를 **$(\mathbf{x},f(\mathbf{x}))$를 지나는 affine function**으로 근사한 것으로 생각할 수 있다. 이 관점에서 subgradient의 정의를 다시 생각해보면, affine function $f(\mathbf{x}) + g^T (\mathbf{z}-\mathbf{x})$가 함수 $f$의 **global underestimator**가 되게 하는 $g$를 subgradient로 받아들일 수 있다. subgradient라는 이름이 붙게 된 어원도 '아래'를 의미하는 'sub-'임을 생각해 볼 수 있다.  
  
### When $f$ is Convex Function

그림을 통해 subgradient에 대해 좀 더 알아보자. 먼저 $f$가 convex한 함수일 때, 아래 그림과 같은 예시를 생각해볼 수 있다.

![image](https://user-images.githubusercontent.com/45325895/52633327-55aa2000-2f07-11e9-96e9-9430524145d1.png){: .center-image}

Convex한 함수 $f$의 미분가능한 점 $x_1$에서는, $f(x_1)+ g_1^T (z-x_1)$이 $f(z)$보다 항상 작거나 같으므로, $x_1$에서의 gradient(그림 상 $g_1$)가 subgradient임을 간단하게 확인할 수 있다. 또한, $(x_1, f(x_1))$에서는, $g_1$이 아닌 다른 기울기의 직선을 그리면 그 직선은 함수 $f$보다 커지는 순간이 생기기 때문에, Convex한 함수 $f$의 미분가능한 점에서는 gradient가 유일한 subgradient임을 알 수 있다. 따라서 미분가능한 점 $x_1$에서의 $f$의 subdifferential, $\partial f(x_1)$은 다음과 같이 gradient를 유일한 원소로 갖는 집합이 된다.

$$
\partial f(x_1)= \{ \nabla f(x_1) \}
$$


Convex한 함수 $f$의 미분불가능한 점 $x_2$에서는 $g_2, g_3$와 같이 $f(x_2)+ g^T (z-x_2) \le f(z)$를 만족하는 벡터 $g$, 즉 subgradient가 무한히 많이 존재할 수 있다. 이 때, $x_2$에서의 $f$의 subdifferential, $\partial f(x_2)$는 무한히 많은 원소를 갖는 집합이 된다.

따라서, $f$가 convex한 함수일 때는, 정의역 내 모든 점 $\forall \mathbf{x} \in \mathbb{R}^n$에 대해 subgradient가 정의되고, subdifferential이 항상 nonempty set이 된다.

### When $f$ is Non-convex Function

함수 $f$가 non-convex 함수일 경우, 경우에 따라 subgradient가 존재할 수도 있고, 존재하지 않을 수도 있다. 모든 점에 대해 subgradient가 존재하지 않는 non-convex 함수의 예시는 아래의 그림과 같다. 함수 $f(x,y)=-(\mid x\mid +\mid y \mid)$의 경우는 모든 점에서 subgradient가 존재하지 않는다. 어느 점에서 어느 기울기의 평면을 그려도 $f$보다 항상 값이 작은 affine function을 그릴 수 없기 때문이다.

![image](https://user-images.githubusercontent.com/45325895/52638636-b93b4a00-2f15-11e9-9541-509c45f0b367.png){: .center-image}
<br>
<br>

# Back to KKT Conditions

KKT 조건은 다음 네 조건을 말한다. 
  
$\text{1. (stationarity) } \enspace 0 \in \partial \Big( f(\mathbf{x})+\sum_{i=1}^{m} u_i h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{x}) \Big)$  
$\text{2. (complementary slackness) } \enspace u_i \cdot h_i(\mathbf{x})=0 \enspace \text{ for all } i$  
$\text{3. (primal feasibility) } \enspace h_i(\mathbf{x}) \le 0, \enspace \ell_j(\mathbf{x}) = 0 \enspace \text{ for all } i,j $  
$\text{4. (dual feasibility) } \enspace u_i \ge 0 \enspace \text{ for all } i$  
  
Subdifferential에 대해 살펴보았으니, 이제 1번 조건이 무엇을 의미하는지 알아본 후, KKT 조건이 strong duality와 어떤 관계를 가지고 있는지 알아보자. 어떤 $\mathbf{x}$에 대해 stationarity 조건이 만족한다면, 

 * 이는, $0 \in \mathbb{R}^n$이 함수 $f(\mathbf{x})+\sum_{i=1}^{m} u_i h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{x})$의 $\mathbf{x}$에서의 subdifferential의 원소임을 의미한다. 
 * 즉, $0 \in \mathbb{R}^n$이 함수 $f(\mathbf{x})+\sum_{i=1}^{m} u_i h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{x})$의 $\mathbf{x}$에서의 subgradient이 됨을 의미한다.

$0 \in \mathbb{R}^n$이 어떤 함수 $f$의 $\mathbf{x}$에서의 subgradient라는 것은 어떤 의미일까? 함수 $f:\mathbb{R}^n \rightarrow \mathbb{R}$에 대해 다음을 만족하는 벡터 $g \in \mathbb{R}^n$를 $\mathbf{x}$에서의 $f$의 *subgradient*라고 한다.

$$
f(\mathbf{z}) \ge f(\mathbf{x}) + g^T (\mathbf{z}-\mathbf{x}) \enspace \text{ for all }\mathbf{z} \in \mathbb{R}^n
$$

이 때, $g =0$이 $\mathbf{x}$에서의 subgradient가 된다면 다음 식이 만족한다.

$$
f(\mathbf{z}) \ge f(\mathbf{x}) \enspace \text{ for all }\mathbf{z} \in \mathbb{R}^n
$$

다시 말해서, **$\mathbf{x}$가 함수 $f$의 minimizer라는 것**을 의미한다. 이를 통해 KKT condition의 stationarity condition은 다음과 같이 이해할 수 있다.  
  
$$
\text{1}' \text{. (stationarity) } \enspace \mathbf{x} = \text{arg}\min_\mathbf z \Big( f(\mathbf{z})+\sum_{i=1}^{m} u_i h_i(\mathbf{z}) +\sum_{j=1}^{r} v_j \ell_j(\mathbf{z}) \Big)
$$

<br>

이제, Strong duality가 성립하는 최적화 문제에 대해 다음이 성립하는 것을 보이자.


$$
\mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ are primal and dual solutions}
$$
$$
\iff \mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ satisfy KKT conditions}
$$




## Necessity

$\mathbf{x}^\ast$와 $\mathbf{u}^\ast,\mathbf{v}^\ast$가 각각 primal, dual solution이라고 하자. 우리는 $\mathbf{x}^\ast$와 $\mathbf{u}^\ast,\mathbf{v}^\ast$에 대해, KKT 조건의 네 조건이 만족하는 것을 보이고자 한다. 현재 우리는 strong duality가 성립하는 문제를 다루고 있으므로, strong duality에 의해 다음이 성립한다.


$$
f(\mathbf{x}^\ast)=g(\mathbf{u}^\ast,\mathbf{v}^\ast)
$$

함수 $g$는 $g(\mathbf{u}, \mathbf{v})=\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}, \mathbf{v})$로 정의되었으므로, 다음이 성립한다.

$$
\begin{align*}
f(\mathbf{x}^\ast)&=g(\mathbf{u}^\ast,\mathbf{v}^\ast)\\
&=\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}^\ast,\mathbf{v}^\ast)
\end{align*}
$$

위 식의 우변은 함수 $L(\mathbf{x}, \mathbf{u}^\ast,\mathbf{v}^\ast)$를 $\mathbf{x}$에 대해 최소화한 값이므로, $L(\mathbf{x}, \mathbf{u}^\ast,\mathbf{v}^\ast)$에 어떤 $\mathbf{x}$를 대입하더라도 그 값은 위 식의 우변보다 크거나 같을 것이다. 따라서, $\mathbf{x}=\mathbf{x}^\ast$를 대입하면 다음과 같다.

$$
\begin{align*}
f(\mathbf{x}^\ast)&=g(\mathbf{u}^\ast,\mathbf{v}^\ast)\\
&=\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}^\ast,\mathbf{v}^\ast)\\
&\le  L(\mathbf{x}^\ast, \mathbf{u}^\ast,\mathbf{v}^\ast)\\
&=f(\mathbf{x}^\ast)+\sum_{i=1}^{m} u_i^\ast h_i(\mathbf{x}^\ast) +\sum_{j=1}^{r} v_j^\ast \ell_j(\mathbf{x^\ast})
\end{align*}
$$

$\mathbf{x}^\ast$는 primal solution이므로, primal constraint를 모두 만족한다. $h_i(\mathbf{x^\ast}) \le 0 , \enspace \ell_j(\mathbf{x^\ast}) = 0 .$ 이를 대입하면, 위 식은 아래와 같이 정리할 수 있다.

$$
\begin{align*}
f(\mathbf{x}^\ast)&=g(\mathbf{u}^\ast,\mathbf{v}^\ast)\\
&=\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}^\ast,\mathbf{v}^\ast)\\
&\le L(\mathbf{x}^\ast, \mathbf{u}^\ast,\mathbf{v}^\ast)\\
&=f(\mathbf{x}^\ast)+\sum_{i=1}^{m} u_i^\ast h_i(\mathbf{x}^\ast) +\sum_{j=1}^{r} v_j^\ast \ell_j(\mathbf{x^\ast})\\
&\le f(\mathbf{x}^\ast)
\end{align*}
$$

이 때, 위 식에서의 부등호들($\le$)은 모두 등호가 만족하게 된다. 따라서 우리는 아래와 같은 두 식을 얻어낼 수 있다.

$$
\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}^\ast,\mathbf{v}^\ast)= L(\mathbf{x}^\ast, \mathbf{u}^\ast,\mathbf{v}^\ast)
$$

$$
u_i^\ast h_i(\mathbf{x}^\ast)=0 \enspace \text{  for all }i
$$

첫 번째 식은 함수 $L(\mathbf{x}, \mathbf{u}^\ast, \mathbf{v}^\ast)=f(\mathbf{x})+\sum_{i=1}^{m} u_i^\ast h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j^\ast \ell_j(\mathbf{x})$의 값이 $\mathbf{x}=\mathbf{x}^\ast$에서 최소가 됨을 의미한다. 이는 $\mathbf{x}=\mathbf{x}^\ast$, 그리고 $\mathbf{u}^\ast, \mathbf{v}^\ast$에 대해, 위에서 보았던 **KKT 조건의 stationarity 조건이 성립**함을 의미한다.

$$
\text{1}' \text{. (stationarity) } \enspace \mathbf{x}^\ast = \text{arg}\min_\mathbf x \Big( f(\mathbf{x})+\sum_{i=1}^{m} u_i^\ast h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j^\ast \ell_j(\mathbf{x}) \Big)
$$

두 번째 식은 $\mathbf{x}^\ast$, 그리고 $\mathbf{u}^\ast, \mathbf{v}^\ast$에 대해, **KKT 조건의 complementary slackness 조건이 성립**함을 의미한다.

$$
\text{2. (complementary slackness) } \enspace u_i^\ast \cdot h_i(\mathbf{x^\ast})=0 \enspace \text{ for all } i
$$

또한, $\mathbf{x}^\ast$와 $\mathbf{u}^\ast,\mathbf{v}^\ast$가 각각 primal, dual solution이므로, $\mathbf{x}^\ast,\mathbf{u}^\ast,\mathbf{v}^\ast$는 primal problem과 Lagrange dual problem의 제약을 당연히 모두 만족하게 된다.

$$
\text{3. (primal feasibility) } \enspace h_i(\mathbf{x}^\ast) \le 0, \enspace \ell_j(\mathbf{x}^\ast) = 0 \enspace \text{ for all } i,j 
$$  

$$
\text{4. (dual feasibility) } \enspace u_i^\ast \ge 0 \enspace \text{ for all } i
$$

<br>

정리하자면, **Strong duality가 성립하는 최적화 문제에 대해,**

$$
\mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ are primal and dual solutions}
$$

$$
\Longrightarrow \mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ satisfy KKT conditions}
$$

<br>


## Sufficiency

우리는 $\mathbf{x}^\ast$와 $\mathbf{u}^\ast,\mathbf{v}^\ast$가 KKT 조건을 만족하는 벡터들일 때, $\mathbf{x}^\ast$와 $\mathbf{u}^\ast,\mathbf{v}^\ast$가 각각 primal과 dual solution이 되며, strong duality도 달성된다는 것을 보이고자 한다. KKT 조건의 stationarity 조건에 의해, 다음이 성립한다.

$$
\mathbf{x}^\ast = \text{arg}\min_\mathbf x \Big( f(\mathbf{x})+\sum_{i=1}^{m} u_i^\ast h_i(\mathbf{x}) +\sum_{j=1}^{r} v_j^\ast \ell_j(\mathbf{x}) \Big)= \text{arg}\min_\mathbf x L(\mathbf{x}, \mathbf{u}^\ast, \mathbf{v}^\ast)
$$

$$
g(\mathbf{u}^\ast, \mathbf{v}^\ast)=\min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}^\ast, \mathbf{v}^\ast)=L(\mathbf{x}^\ast, \mathbf{u}^\ast, \mathbf{v}^\ast)
$$

$\mathbf{x}^\ast$와 $\mathbf{u}^\ast$는 KKT 조건의 complementary slackness 조건을 만족하므로, $u_i^\ast \cdot h_i(\mathbf{x^\ast})=0  \text{ for all } i.$ 또한, $\mathbf{x}^\ast$는 KKT 조건의 primal feasibility 조건을 만족하므로, $\ell_j(\mathbf{x^\ast})=0 \text{ for all } j.$ 따라서 이를 대입하면 다음 식을 얻는다.

$$
\begin{align*}
g(\mathbf{u}^\ast, \mathbf{v}^\ast)&=f(\mathbf{x}^\ast)+\sum_{i=1}^{m} u_i^\ast h_i(\mathbf{x}^\ast) +\sum_{j=1}^{r} v_j^\ast \ell_j(\mathbf{x^\ast})\\
&=f(\mathbf{x}^\ast) \enspace \enspace \text{ : strong duality}
\end{align*}
$$

$\mathbf{x}^\ast$와 $\mathbf{u}^\ast,\mathbf{v}^\ast$가 KKT 조건을 만족할 때, strong duality가 성립한다. 다시 말해서, $\mathbf{x}^\ast$와 $\mathbf{u}^\ast,\mathbf{v}^\ast$가 각각 primal, dual solution이 된다는 사실을 증명한 것이다. Necessity를 보일 때는 이 최적화 문제가 strong duality를 만족하는 문제라는 사실을 가정하고 necessity가 성립함을 보였다. 하지만 지금은 그러한 가정 없이 임의의 최적화 문제에 대해, KKT 조건을 만족하는 벡터들 $\mathbf{x}^\ast, \mathbf{u}^\ast,\mathbf{v}^\ast$가 primal, dual solution이 된다는 사실과, strong duality가 만족한다는 사실을 보인 것이다.  
<br>

정리하자면, **임의의 최적화 문제에 대해,**

$$
\mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ satisfy KKT conditions}
$$

$$
\Longrightarrow \mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ are primal and dual solutions, and strong duality holds}
$$

<br>

## Conclusion

Strong duality가 성립하는 최적화 문제에 대해,
$$
\mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ are primal and dual solutions}
$$

$$
\Longrightarrow \mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ satisfy KKT conditions}
$$

 임의의 최적화 문제에 대해,

$$
\mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ satisfy KKT conditions}
$$

$$
\Longrightarrow \mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ are primal and dual solutions, and strong duality holds}
$$

따라서, Strong duality가 성립하는 최적화 문제에 대해서

$$
\mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ are primal and dual solutions}
$$

$$
\iff \mathbf{x}^\ast \text{ and } \mathbf{u}^\ast,\mathbf{v}^\ast \text{ satisfy KKT conditions}
$$

(최적화 문제가 strong duality를 만족하게 하는 조건의 대표적인 예는 위에서 소개한 Slater's condition이 있다.)
