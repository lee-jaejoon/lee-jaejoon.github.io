---
layout: post
title: "The Bootstrap Method"
tags: [Statistics]
comments: true
---
> Contents  
> [1. Introduction](#1-introduction)  
> [2. Theoretical Background of Bootstrap Method](#2-theoretical-background-of-bootstrap-method)  
> [3. R Example](#3-r-example)  
> [4. Application](#4-application)  
  


# 1. Introduction

넓은 의미에서 Bootstrap은 random sampling with replacement에 기반한 통계 검정 혹은 추정을 의미하며, 일반적으로는 특정 통계량(추정량)의 분포를 구하기 위해, **관측된 random sample에서 resampling을 수행하는 case resampling 기법**을 의미한다. 이 포스트에서는 간단한 예시와 함께 bootstrap의 이론적인 기반을 설명하고, bootstrap이 주로 사용되는 적용 사례들을 소개함으로써 bootstrap에 대해 보다 깊이 알아보고자 한다.

bootstrap이 가장 흔하게 사용되는 예시로는, analytic하게 통계량의 분포를 구할 수 없는 경우, 그 통계량의 분포를 추정하기 위해 bootstrap을 사용하는 경우이다. 예를 들어, 다음과 같이 표준정규분포로부터 $100$개의 관찰치를 갖는 iid random sample을 뽑는 경우를 생각해보자. 

$$
X_1, X_2, \cdots , X_{100} \sim iid \enspace N(0,1)
$$

표본평균 $\bar X$와 같이 간단한 통계량의 경우, 그 분포를 analytic하게 바로 도출할 수 있다.

$$
\bar X = \frac{X_1+ X_2+ \cdots +X_{100}}{100} \sim \enspace N(0,\frac{1}{100})
$$

그러나 우리의 관심사가 random sample에 대한 복잡한 통계량의 분포라면 어떻게 할까? 혹은 우리가 모집단의 분포에 대해 알지 못한다면 어떻게 할 것인가? 예를 들어 모집단의 분포를 알지 못하는 random sample, $X_1, X_2, \cdots , X_{100}$에 대해, $10^{th}$ order statistic, $Y$의 분포(sampling distribution)를 구하고자 하는 경우를 생각해보자. 이 때는 통계량도 복잡하고 모집단의 분포도 알지 못하므로, 위에서 표본평균의 분포를 구했던 것과 같이 그 분포를 analytic하게 도출할 수 없다.  

사실 우리는 주어진 random sample의 $10^{th}$ order statistic의 값을 구할 수 있다. Random sample, $X_1, X_2, \cdots , X_{100}$를 값의 크기 오름차순으로 정렬한 후, 열 번째에 오는 관찰치의 값이 주어진 random sample의 $10^{th}$ order statistic의 값인 것이다. 하지만 이는 하나의 값, 하나의 점 추정치(point estimate)이기 때문에 우리에게 도움이 되지 않는다. 우리는 100개의 random sample을 뽑았을 때, $10^{th}$ order statistic, $Y$의 **분포**를 알아내는 것이 목표이기 때문이다.  

Bootstrap은 다음과 같은 방법으로 $10^{th}$ order statistic(from size $100$ random sample)의 분포를 추정한다.

 * 관측된 random sample, $X_1, X_2, \cdots , X_{100}$로부터, $100$개의 관찰치를 갖는 random sample을 복원추출로 $B$개 생성한다. 다시 말해서, 주어진 random sample, $X_1, X_2, \cdots , X_{100}$로부터 $100$개를 복원추출하는 작업을 $B$번 수행하는 것이다. Sample에서 다시 sample을 뽑는 작업이기 때문에, 이를 **resampling**이라고 부르며, 그 resample의 결과로 만들어진 $B$개의 sample들을 **bootstrap sample**이라고 한다.

 * random sample로부터 뽑은 bootstrap sample $B$개에 대해, 각각 $10^{th}$ order statistic의 값, $y^{\ast1}, \cdots, y^{\ast B}$을 모두 구한다.

$$

(X_1, X_2, \cdots , X_{100}) \enspace  \stackrel{\text{resampling}}{\Rightarrow } \enspace 

{\begin{array}{c}
(X_1^{\ast1}, X_2^{\ast1}, \cdots , X_{100}^{\ast1}) \rightarrow y^{\ast1}\\
(X_1^{\ast2}, X_2^{\ast2}, \cdots , X_{100}^{\ast2}) \rightarrow y^{\ast2} \\
\vdots \\
(X_1^{\ast B}, X_2^{\ast B}, \cdots , X_{100}^{\ast B}) \rightarrow y^{\ast B} \\
\end{array} }

$$

 * $B$개의 bootstrap sample들에서 각각 구한 $10^{th}$ order statistic의 값, $y^{\ast1}, \cdots, y^{\ast B}$의 히스토그램을 그린다. 이 히스토그램이 나타내는 분포가, $100$개의 random sample을 뽑았을 때의 $10^{th}$ order statistic, $Y$의 분포를 bootstrap을 이용해 추정한 것이다.

![image](https://user-images.githubusercontent.com/45325895/52214067-1f9be900-28d4-11e9-893a-3d58c2770e89.png){:. center-image}


우리는 analytic하게는 구할 수 없었던 통계량 $Y$의 분포를 bootstrap을 통해 얻은 히스토그램으로 통째로 추정해버렸으므로, $Var(Y)$가 궁금하다면 $y^{\ast1}, \cdots, y^{\ast B}$의 표본분산을 구하면 되고, $Y$의 평균이 궁금하다면 $y^{\ast1}, \cdots, y^{\ast B}$의 표본평균을 구하면 된다. 
<br>
<br>

# 2. Theoretical Background of Bootstrap Method

Bootstrap은 위에서 본 것과 같이 대단히 쉽고 만능인 것처럼 보인다. 그러나 bootstrap은 그 적용에 있어 주의해야 할 부분이 있고, 그를 위해서는 bootstrap을 통한 통계량의 분포 추정이 왜 가능한 것인지를 알아야 한다. 위의 예시에서 우리의 목표는 random sample $X_1, X_2, \cdots , X_{100}$의 $10^{th}$ order statistic, $Y$의 분포(sampling distribution)를 구하는 것이었다. 좀더 구체적으로 여기서는 random sample로부터 만들어진 통계량 $Y$의 **분산**, $Var(Y)$을 구하는 것을 목표로 설정하자.  

실제로 $Y$가 생성되는 generating process를 따라가보면 아래와 같다. 
1. random sample의 모집단의 분포 $F$로부터 $100$개의 sample, $X_1, X_2, \cdots , X_{100}$을 뽑고, 

2. 그 100개의 random sample observation 중 $10^{th}$ order statistic을 $Y$로 설정하는 것이다.

$$
F \enspace  \stackrel{\text{random sampling}}{\Rightarrow } \enspace \mathbf{X}=(X_1, X_2, \cdots , X_{100}) \enspace  \stackrel{Y=g(\mathbf{X})=X_{(10)}}{\Rightarrow } \enspace Y
$$

$$
\text{unknown population dist. } \enspace \enspace \enspace  \text{observed random sample} \enspace\enspace  \enspace \enspace \enspace  \text{statistic of interest}\enspace \enspace 
$$

그리고 이와 같은 과정을 모든 가능한 random sample의 경우에 대해 다 고려하여 분산을 구한 것이, 우리가 구하고자 하는 $Var(Y)$이 생성되는 과정일 것이다.  

**Bootstrap은 이 과정을 모방하는 것을 목표로 한다.** 하지만 우리는 모집단의 분포 $F$를 모르기 때문에, 모집단으로부터 random sample을 뽑을 수 없다. 그래서 그 대신, **empirical distribution**, $\hat F$로부터 random sample을 뽑는다. 

## Empirical Distribution Function

Empirical distribution이란 $n$개의 데이터, $X_1, X_2, \cdots , X_n$이 있을 때, 각 data point에서 $\frac{1}{n}$씩 density를 갖는 이산확률분포이다. 이는 주로 누적분포함수의 형태로 나타내는데, empirical distribution function은 각 데이터 점에서 $\frac{1}{n}$씩 상승하는 계단 모양의 step function이다. 

![image](https://user-images.githubusercontent.com/45325895/52216666-1f9ee780-28da-11e9-8739-78eb73094d4c.png){: .center-image}

위 그림은 표준정규분포에서 생성된 $20$개의 관찰치를 갖는 random sample에 대한 empirical distribution function를 그린 것이다. 주황색 곡선으로 나타난 곡선은 표준정규분포의 실제 누적분포함수이다. 모집단의 실제 확률분포함수는 아니지만, 관측된 random sample을 통해 경험적으로(empirically) 얻어진 확률분포함수라는 뜻에서 경험적(empirical) 분포 함수라는 이름이 붙여졌다. 만약 random sample의 크기를 $20$에서 더 늘리면 empirical distribution function은 어떻게 될까? 다음 그림은 random sample의 크기를 $160$으로 늘렸을 때의 empirical distribution function이다.

![image](https://user-images.githubusercontent.com/45325895/52217255-6b05c580-28db-11e9-9e0b-852aa79b68c1.png){: .center-image}


위 그림에서 확인할 수 있는 것과 같이, empirical distribution function $\hat F$은 random sample의 size($n$)가 커질 수록, 실제 모집단의 확률분포함수 $F$에 (almost surely) 수렴하는 것이 알려져있다. 다만 이 때 empirical distribution function $\hat F$이 모집단의 확률분포 $F$에 수렴하려면, empirical distribution이 기반한 **sample이 모집단으로부터 IID 가정 하에 뽑힌 random sample이어야 한다**. empirical distribution function의 asymptotic behavior에 대한 보다 자세한 내용이 궁금하다면 **[Glivenko-Cantelli 정리](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem)**를 확인하자.  
<br>
<br>

다시 원래의 이야기로 돌아와서, 우리는 bootstrap을 통해 $Y$가 생성되는 generating process를 모방하여, $Y$의 분산, $Var(Y)$을 구하는 것이 목표였다. 그래서 모집단의 분포 $F$에서 random sample을 뽑는 대신, empirical distribution $\hat F$에서 random sample을 뽑아 bootstrap sample을 만들고, 그를 통해 관심 통계량이었던 $Y$의 복제품이라고 할 수 있는 $Y^\ast$를 만든다. 이를 모식도로 나타내면 아래와 같다.

 * In the real world,

$$
F \enspace  \stackrel{\text{random sampling}}{\Rightarrow } \enspace \mathbf{X}=(X_1, X_2, \cdots , X_{100}) \enspace  \stackrel{Y=g(\mathbf{X})=X_{(10)}}{\Rightarrow } \enspace Y
$$

$$
\text{unknown population dist. } \enspace \enspace \enspace  \text{observed random sample} \enspace\enspace  \enspace \enspace \enspace  \text{statistic of interest}\enspace \enspace 
$$

 * In the bootstrap world,

$$
\hat F \enspace  \stackrel{\text{random sampling}}{\Rightarrow } \enspace \mathbf{X}^\ast=(X_1^\ast, X_2^\ast, \cdots , X_{100}^\ast) \enspace  \stackrel{Y^\ast=g(\mathbf{X}^\ast)=X^\ast_{(10)}}{\Rightarrow } \enspace Y^\ast
$$

$$
\enspace \enspace  \text{empirical distribution } \enspace \enspace \enspace \enspace \enspace \enspace  \text{bootstrap sample} \enspace\enspace  \enspace \enspace \enspace \enspace \enspace  \enspace \text{bootstrap replication}\enspace \enspace 
$$

Introduction의 예시에서, 모집단으로부터 뽑힌 random sample로부터 복원추출로 bootstrap resampling을 해서 $Y^\ast$를 구했던 것은, 바로 empirical distribution으로부터 iid random sample을 뽑아야 했기 때문이었던 것이다.  

그런데 우리는 통계량 $Y$를 replicate하는 것이 목표가 아니라, $Y$의 분산을 추정하는 것이 목표이다. 통계량 $Y=g(\mathbf{X})$$=g(X_1, X_2, \cdots , X_{100})$의 분산, $Var(Y)$은 다음과 같이 나타낼 수 있다. 여기서 $dF_j$는 $dF_j=f_j(X_j)dX_j=f(X_j)dX_j$를 나타낸다.(CDF를 확률변수로 미분하면 PDF인 것을 떠올리면, notation을 이해하기 쉽다 : $\frac{dF_X(x)}{dx}=f_X(x)$)

$$
Var(Y)=Var(g(\mathbf{X}))=\mathbb{E}_{F_1 \cdots F_{100}}
\Bigg[ \Big[ g(X_1, X_2, \cdots , X_{100})-E[g(X_1, X_2, \cdots , X_{100})] \Big]^2 \Bigg]
$$

$$
=\int \cdots \int \Big[ g(X_1, X_2, \cdots , X_{100})-E[g(X_1, X_2, \cdots , X_{100})] \Big]^2 dF_1 \cdots dF_{100}
$$

위 모식도에서 볼 수 있듯이, Bootstrap은 모집단의 누적분포함수 $F$를 empirical distribution function $\hat F$으로 대체했으므로, bootstrap으로 얻은 $Var(Y)$의 추정치는 다음과 같다. 이를 **이상적 bootstrap 추정치**, **ideal bootstrap estimate**이라고 한다. 적분 내의 $d\hat F_j$는 $d\hat F_j=\hat f(X_j^\ast)dX_j^\ast$를 의미한다.

$$
\hat {Var}_{ideal}(Y)=\mathbb{E}_{\hat F_1 \cdots \hat F_{100}}
\Bigg[ \Big[ g(X_1^\ast, X_2^\ast, \cdots , X_{100}^\ast)-E[g(X_1^\ast, X_2^\ast, \cdots , X_{100}^\ast)] \Big]^2 \Bigg]
$$

$$
=\int \cdots \int \Big[ g(X_1^\ast, X_2^\ast, \cdots , X_{100}^\ast)-E[g(X_1^\ast, X_2^\ast, \cdots , X_{100}^\ast)] \Big]^2 d\hat F_1 \cdots d\hat F_{100}
$$

그런데 bootstrap sample, $X_1^\ast, X_2^\ast, \cdots , X_{100}^\ast$가 생성된 분포, $\hat F$를 안다고 하더라도, random sample의 크기 $n$이 작지 않은 이상, 이 적분의 값을 계산하는 것은 쉽지 않은 경우가 많다. 그래서 우리는 **ideal bootstrap estimate**을 그대로 사용하는 것이 아니라, 이 **ideal bootstrap estimate**를 **Monte Carlo Integration**으로 한 번 더 근사한 것을 bootstrap 추정의 최종 결과로 사용한다.

## Monte Carlo Integration

Monte Carlo integration은 적분의 값을 근사를 통해 구하는 numerical 방법 중 하나이며, 난수 생성을 이용하여 적분값 근사를 수행한다. 적분구간 $D$ 위에서 함수 $f$를 적분한 값, $F$를 구하고 싶다고 해보자.

$$
F=\int_D f(x)dx
$$

$D$를 support로 가지며, 난수를 생성할 수 있는 확률분포함수 $p(x)$를 가지고 있다고 하자. 이를 이용하여 위 적분식을 아래와 같이 바꿀 수 있다. 이는 $\frac{f(X)}{p(X)}$를, $p(x)$의 확률분포함수를 갖는 확률변수에 대해 기대값을 취한 것과 같다.

$$
F=\int_D \frac{f(x)}{p(x)}p(x)dx=\mathbb{E}_p \Big[ \frac{f(X)}{p(X)} \Big]
$$

대수의 법칙은 $N$이 충분히 큰 random sample로부터 만든 표본평균이 모평균으로 확률수렴하는 것을 보장한다. 따라서, 위 식을 아래와 같이 근사할 수 있다.

$$
F=\mathbb{E}_p \Big[ \frac{f(X)}{p(X)} \Big] \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)}
$$

따라서, $p(x)$를 확률분포함수로 갖는 난수를 $N$개 생성한 뒤, 생성된 난수, $x_1,x_2, \cdots , x_N$에 대해 $\frac{f(x)}{p(x)}$의 값을 각각 모두 구한 뒤, 이를 표본평균내는 방법으로 $F$의 적분값을 추정한다.

<br>
<br>

Monte Carlo integration을 이용해서 ideal bootstrap estimate을 추정한 $Y$의 분산에 대한 bootstrap estimate은 다음과 같다.

$$
\mathbb{E}_{\hat F_1 \cdots \hat F_{100}}
\Bigg[ \Big( g(\mathbf{X}^\ast)-E[g(\mathbf{X}^\ast)] \Big)^2 \Bigg]
\approx
\frac{1}{B} \sum_{i=1}^{B} \Big( g(\mathbf{X}^\ast_i)-E[g(\mathbf{X}^\ast_i)] \Big)^2
$$

$$
\text{where }\enspace \mathbf{X}^\ast_i=(X^\ast_{i1},X^\ast_{i2},\cdots,X^\ast_{i,100}) \text{ : } i \text{ th bootstrap sample} \enspace \sim \enspace \hat F \text{ : empirical distribution}
$$

즉, 관측된 random sample에서 resampling을 통해, empirical distribution에서 bootstrap sample을 $B$회 뽑고, 각각 $Y=g(\mathbf{X})$의 편차 제곱, $( g(\mathbf{X}^\ast)-E[g(\mathbf{X}^\ast)] )^2 $의 값을 구하여 표본평균을 구한 것이다. 이는 위에서 수행한 다음 작업과 같다.

 > $B$개의 bootstrap sample들에서 각각 구한 $10^{th}$ order statistic의 값, $y^{\ast1}, \cdots, y^{\ast B}$의 히스토그램을 그린다. 이 히스토그램이 나타내는 분포가, $100$개의 random sample을 뽑았을 때의 $10^{th}$ order statistic, $Y$의 분포를 bootstrap을 이용해 추정한 것이다.  
  
<br>

## Review of Bootstrap Process

지금까지 설명한 bootstrap의 수행과정과 이론적인 근거를 정리하면 다음과 같다. 관측된 sample, $\mathbf{X}=(X_1,X_2,\cdots,X_N)$로부터 bootstrap sample을 $B$회 생성한다고 하자. 우리의 목표는 통계량 $Y=g(\mathbf{X})=g(X_1,X_2,\cdots,X_N)$의 분산을 추정하는 것이다.
<br>

$$
Var(g(\mathbf{X})) \stackrel{\text{empirical distribution } \hat F}{\Rightarrow } \hat {Var}_{ideal}(Y) \stackrel{\text{Monte Carlo Integration}}{\Rightarrow } \hat {Var}(Y)
$$

<br>

$$
Var(Y)=\mathbb{E}_{F_1 \cdots F_{100}}
\Big[ \Big( g(\mathbf{X})-E[g(\mathbf{X})] \Big)^2 \Big] \text{ : target statistic}
$$

$$
\Downarrow \enspace \enspace \enspace \text{by Glivenko-Cantelli thm.} 
$$

$$
\hat {Var}_{ideal}(Y)=\mathbb{E}_{\hat F_1 \cdots \hat F_{100}}
\Big[ \Big( g(\mathbf{X}^\ast)-E[g(\mathbf{X}^\ast)] \Big)^2 \Big] \text{ : ideal bootstrap estimate}
$$

$$
\Downarrow \enspace \enspace \enspace \text{by law of large numbers}
$$

$$
\hat {Var}(Y) = \frac{1}{B} \sum_{i=1}^{B} \Big( g(\mathbf{X}^\ast_i)-E[g(\mathbf{X}^\ast_i)] \Big)^2 \text{ : bootstrap estimate}
$$

<br>

 * 주어진 random sample, $X_1, X_2, \cdots , X_N$로부터 $N$개를 복원추출(resampling)하는 작업을 $B$번 수행한다. 그 resample의 결과로 만들어진 $B$개의 sample들을 **bootstrap sample**이라고 한다.

	 * $Y$는 모집단 분포 $F$에서 생성된 random sample, $\mathbf{X}$에 대한 함수이므로, 이를 모방하기 위해 empirical distribution $\hat F$로부터 bootstrap sample을 생성하는 것. empirical distribution은 $N$이 커짐에 따라 모집단의 분포에 수렴한다는 성질이 있다.(Glivenko-Cantelli 정리)

 * random sample로부터 뽑은 bootstrap sample $B$개에 대해, 각각 $10^{th}$ order statistic의 값, $y^{\ast1}, \cdots, y^{\ast B}$을 모두 구한다.

$$

(X_1, X_2, \cdots , X_{100}) \enspace  \stackrel{\text{resampling}}{\Rightarrow } \enspace 

{\begin{array}{c}
(X_1^{\ast1}, X_2^{\ast1}, \cdots , X_{100}^{\ast1}) \rightarrow y^{\ast1}\\
(X_1^{\ast2}, X_2^{\ast2}, \cdots , X_{100}^{\ast2}) \rightarrow y^{\ast2} \\
\vdots \\
(X_1^{\ast B}, X_2^{\ast B}, \cdots , X_{100}^{\ast B}) \rightarrow y^{\ast B} \\
\end{array} }

$$

 * $B$개의 bootstrap sample들에서 각각 구한 $10^{th}$ order statistic의 값, $y^{\ast1}, \cdots, y^{\ast B}$의 히스토그램을 그린다. 이 히스토그램이 나타내는 분포가, $100$개의 random sample을 뽑았을 때의 $10^{th}$ order statistic, $Y$의 분포를 bootstrap을 이용해 추정한 것이다.

	 * 이는 ideal bootstrap 추정치를 Monte Carlo integration으로 추정하는 것과 같다. $Y^\ast$의 **편차 제곱의 기대값**($Var(Y)$)을 $Y^\ast$의 **편차 제곱의 표본평균**($\hat {Var}(Y)$)으로 추정하는 것. 이는 대수의 법칙에 의해 뒷받침된다.  
<br>
<br>
<br>

# 3. R Example
평균이 $15$인 지수분포(exponential distribution)에서 생성한 $20$개의 iid random sample에 대해, bootstrap resampling을 수행하여 표본평균의 분포를 근사해보자.
```r
set.seed(123)
sample <- rexp(20, 1/15)
mean(sample)
```
![image](https://user-images.githubusercontent.com/45325895/52261221-fe86d700-296b-11e9-8138-0c3ec5de27d3.png)

평균이 $15$인 지수분포에서 $20$개의 random sample을 뽑았지만, 그 표본평균은 $15$보다 많이 낮은 $12.16759$가 나왔다.
```r
sample
```
![image](https://user-images.githubusercontent.com/45325895/52261288-47d72680-296c-11e9-8dba-96098c2fb09c.png)

그 이유는 sample내의 $0.4736604$, $0.8431646$, $0.4373017$의 영향으로 판단된다. 우연히 값이 현저하게 낮은 값들이 많이 포함된 sample이 생성된 것이다.
```r
pexp(0.4736604, 1/15)
pexp(0.8431646, 1/15)
pexp(0.4373017, 1/15)
```
![image](https://user-images.githubusercontent.com/45325895/52261384-9258a300-296c-11e9-8eac-d4f4185f0454.png)

$20$개의 관찰치를 갖는 random sample 중, $6$th percentile보다 작은 값이 세 개나 나왔기 때문에 표본평균의 값이 $15$보다 크게 낮은 값이 나온 것이다. 이제 bootstrap을 수행하기 위해, 200개의 bootstrap sample을 생성하고, 각 bootstrap sample로부터 표본평균의 값을 구해 히스토그램을 그려 보았다.
```r
result <- rep(0,200)
for(i in 1:200){
  boot.sample <- sample(sample, 20, replace=T)
  result[i] <- mean(boot.sample)
}
#histogram과 true density 그리기
hist(result,breaks=seq(0,100, 1.5),xlim=c(0,30))
par(new=T)
plot(seq(0,100,0.001), dgamma(seq(0,100,0.001), 20, scale = 15/20),cex=0.5,xlim=c(0,30),xlab="",ylab="", axes=F)
```
![image](https://user-images.githubusercontent.com/45325895/52261768-bff21c00-296d-11e9-981c-0194ba5134d8.png){: .center-image}

원래 이론적으로는, 평균이 15인 20개의 지수분포의 표본평균은 $Gamma(20,\frac{15}{20})$를 따른다. 히스토그램 위에 그려진 곡선은 이 $Gamma(20,\frac{15}{20})$의 true density를 그린 것이다. 그러나 200개의 bootstrap sample로 얻은 히스토그램은 true density보다 작은 값으로 치우쳐 있는 것을 확인할 수 있다. 이는 처음 우리가 생성한 random sample이 낮은 값으로 치우친 random sample이었기 때문에, 그로부터 resampling을 수행한 bootstrap의 결과 역시 낮은 값으로 치우치게 된 것이다. 이는 원래의 random sample이 모집단을 잘 반영하지 못한다면, 그 random sample의 empirical distribution을 이용하는 bootstrap의 추정 결과 또한 그럴 확률이 높다는 것을 의미한다. 따라서 이는 bootstrap을 적용함에 있어서 주의해야할 점이라고 할 수 있다.  

그러나 이 예제의 경우 random sample의 크기가 20으로 매우 작은 크기의 sample이었다. 위에서 본 것과 같이, Random sample의 크기 $N$이 더 큰 상황에서는 empirical distribution이 모집단의 누적분포함수에 수렴할 것이므로 이 예제에서와 같은 문제가 발생할 가능성은 더 낮아질 것으로 예상할 수 있다.  
<br>
<br>
<br>




# 4. Application
## Bootstrap Standard Error of an Estimator/Test Statistic
위에서 소개한 것과 같이, Bootstrap은 통계량의 분포를 analytic하게 구할 수 없는 상황에서, random sample이 생성되는 과정을 모방함으로써, 통계량의 분포를 근사할 수 있게 해주는 대안이 된다. 이는 식의 도출로 analytical 표준오차(standard error)를 구하는 것이 불가능한 추정량(estimator) 및 검정통계량(test statistic)의 표준오차 및 신뢰구간을 구할 수 있게 해준다.

## Bootstrap Prediction Error
모형의 선택(Selection)과 평가(Assessment)를 위해서, 우리는 prediction error, 혹은 test error라고도 불리는 extra-sample error를 추정해야한다. 통계 모형 역시 true data generating function을 추정하는 추정량이기 때문에, 큰 틀에서는 3.1에서 소개한 것과 같은 맥락이다. 하지만 단순히 bootstrap 후 loss의 평균을 구하는 것이 아니라, $.632$ estimator, $.632+$ estimator와 같이 test error를 정확히 추정하기 위한 방법들이 연구되어있다. 이들은 실제로 test error를 구하는 데 있어서, cross validation과 비슷한 수준의 성능을 보여주는 것으로 알려져 있다.

## Bagging
**B**ootstrap **Agg**regat**ing**, 부트스트랩 합치기. 위에서와 같이 모수 추정이나 예측의 정확성을 평가하기 위한 목적이 아니라, 추정 혹은 예측 자체의 성능을 개선시키는데 bootstrap을 사용하는 방법이다. 주어진 training data로부터 bootstrap sample을 resampling한 후, 각 bootstrap sample 내에서 fit된 모형들의 예측을 평균내는 방식으로 이루어진다. (classification인 경우는 다수결 원칙으로 결정.) Bagging에 대한 자세한 내용은 The Elements of Statistical Learning의 chapter 8을 정리한 이 링크의 포스트에서 자세히 다루도록 하겠다. **[Link](https://lee-jaejoon.github.io/ESL-8/#87-bagging)**

<br>
<br>
<br>





# Reference
 > Hastie, T., Tibshirani, R.,, Friedman, J. (2001). The Elements of Statistical Learning. New York, NY, USA: Springer New York Inc..  
 > http://galton.uchicago.edu/~eichler/stat24600/Handouts/bootstrap.pdf
