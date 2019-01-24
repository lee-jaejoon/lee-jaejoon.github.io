---
layout: post
title: "Understanding Cross-entropy Cost Function"
tags: [Deep Learning]
comments: true
---
Deep learning을 처음 공부하다 보면 cross-entropy 혹은 cross-entropy cost function를 자주 접하게 된다. 이는 통계학에서 쓰이는 Maximum likelihood 개념과 관계가 있는데,   

대부분의 neural network 모델은 Maximum likelihood를 이용하여 훈련(train)된다. 이 때, 최적 parameter 값 결정을 위해 최소화하고자 하는 target 함수, 즉 cost function은 negative log-likelihood가 된다. 다시 말해서, Cost function인 **negative log-likelihood를 최소화**하는 것은 **maximum likelihood를 달성하는 parameter의 값을 찾는 것**이다. 이처럼 negative log-likelihood를 이용한 cost function은 어떤 의미가 있는지 간단히 살펴보자.  
<br>
<br>

## Why Is It Called "Cross-entropy Cost Function"?
[Cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy)는 원래 정보 이론(Information theory)에서 쓰이는 개념인데, 여기서는 cross-entropy의 정보 이론적 의미를 설명하는 것이 아니라, cross-entropy와 KL Divergence의 비교를 통해 likelihood로 만든 cost function과 어떤 관계가 있는지를 알아보고자 한다. entropy 혹은 cross-entropy의 정보 이론에서의 의미는 다음에 기회가 되면 별도의 포스트로 정리해볼 생각이다. 우선은 두 확률분포 $p$와 $q$에 대하여, cross-entropy가 다음과 같은 식을 의미한다는 것을 알아두자. 

$$
H(p,q)=\mathbb{E}_p [-\log q]
$$

흔히 **negative log-likelihood**로 정의된 cost function은 **cross-entropy cost function**이라고 부른다. 왜 이런 이름이 붙게 되었는지 negative log-likelihood를 살펴보자. 관측된 $N$개의 점을 갖는 Training data 집합 $\mathbb{X}$에 대해, negative log-likelihood를 이용한 parameter 추정 결과는 아래와 같다. 이 때, $p_{model}$은 우리가 만든 모형의 pdf이다.

$$
\hat{\theta}_{MLE} 
= \arg \min _\theta \enspace - p_{model}(\mathbb{X}:\theta) 
= \arg \min _\theta \enspace - \prod_{i=1}^{N}  p_{model}(y_i \mid x_i,\theta)
$$

$$
= \arg \min _\theta \enspace - \sum_{i=1}^{N}  \log p_{model}(y_i \mid x_i,\theta)
= \arg \min _\theta \enspace  - \frac{1}{N} \sum_{i=1}^{N}  \log p_{model}(y_i \mid x_i,\theta)
$$

여기서 training data로 정의된 empirical distribution을 $\hat p_{data}$라고 한다면, 위 식은 아래와 같이 나타낼 수 있다. empirical distribution, $\hat p_{data}$는 training data의 각 점에서 $\frac{1}{N}$의 확률을 갖는 확률질량함수이다.

$$
= \arg \min _\theta \enspace - \mathbb{E}_{x,y \sim \hat{p}_{data}}  \log p_{model}(y_i \mid x_i,\theta) 
= \arg \min _\theta \enspace H(\hat{p}_{data},p_{model})
$$

즉, **negative log-likelihood로 정의한 cost function**은 training data의 empirical distribution, $\hat p_{data}$와 우리가 만든 모형의 분포, $p_{model}$의 **cross-entropy**와 같음을 알 수 있다. 그래서 likelihood를 이용하여 만든 cost function을 **cross-entropy cost function**이라고 부르는 것이다.  
<br>
<br>

## Maximum Likelihood vs KL Divergence
cross-entropy cost function을 최소화하는 것은 우리의 모델 $p_{model}$의 likelihood를 최대화하는 것과 같다는 것을 확인했다. 추가로, cross entropy cost function 혹은 negative log-likelihood cost function을 더 깊게 이해하기 위해, 이것이 **KL Divergence**와는 어떤 관계가 있는지 알아보자. [Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)는 **한 확률분포가 다른 확률분포와 서로 얼마나 다른지를 측정하는 척도**이다. 두 확률분포가 완전히 같다면 KL Divergence는 $0$이 된다. 그 식은 다음과 같다.

$$
D_{KL} (P \mid \mid Q)=- \sum_{x} P(x) \log ( \frac{Q(x)}{P(x)} ) =-\mathbb{E}P \log(\frac{Q(x)}{P(x)})
$$

training data의 empirical distribution, $\hat p_{data}$와 우리가 만든 모형의 분포, $p_{model}$ 사이의 KL Divergence를 구하면 아래와 같다. 우리는 $N$개의 점으로 이루어진 training data가 이미 주어진 것으로 생각하므로, 어떤 모형 $p_{model}$이 최적일지, 그리고 $\theta$가 어떤 값을 가져야 $p_{model}$이 주어진 training data를 잘 설명할지를 고민하는 현재의 상황에서, $ \hat p_{data}$는 변하지 않는 주어진 분포이다. 따라서 아래의 식에 표기할 때 $\theta$에 대한 dependence를 주지 않았다.

$$
D_{KL} (\hat{p}_{data} \mid \mid p_{model})
= - \mathbb{E}_{x,y \sim \hat{p}_{data}}  \log \Big( \frac{p_{model}(y_i \mid x_i,\theta)}{\hat{p}_{data}(y_i \mid x_i)} \Big)
$$

$$
= - \mathbb{E}_{x,y \sim \hat{p}_{data}}  \log \big( p_{model}(y_i \mid x_i,\theta) \big) + \mathbb{E}_{x,y \sim \hat{p}_{data}} \log \big( \hat{p}_{data}(y_i \mid x_i) \big)
$$

$$
= - \mathbb{E}_{x,y \sim \hat{p}_{data}}  \log \big( p_{model}(y_i \mid x_i,\theta) \big) + constant
$$

Likelihood를 maximize하는 것은 우리가 만든 모형의 분포 $p_{model}$이 training set의 분포 $\hat p_{data}$를 가장 잘 설명하도록 만드는 parameter, $\theta$를 고르는 일이다. 쉽게 말해서, $p_{model}$과 $\hat p_{data}$의 차이를 최소화하고 싶다는 것이다. KL Divergence는 두 확률분포, $\hat p_{data}$와 $p_{model}$가 얼마나 서로 다른지를 나타낸다. 따라서, **Likelihood를 maximize하는 $\theta$를 고르는 것**은 $\hat p_{data}$와 $p_{model}$ 사이의 **KL Divergence를 최소화하는 $p_{model}$을, KL Divergence를 최소화하는 $\theta$를 고르는 것**이 된다. 이는 식으로도 확인할 수 있다.  



$$
\arg \min _\theta \enspace D_{KL} (\hat{p}_{data} \mid \mid p_{model}) = \arg \min _\theta \enspace  - \mathbb{E}_{x,y \sim \hat{p}_{data}}  \log \big( p_{model}(y_i \mid x_i,\theta) \big) + constant
$$

$$
= \arg \min _\theta \enspace  - \mathbb{E}_{x,y \sim \hat{p}_{data}}  \log \big( p_{model}(y_i \mid x_i,\theta) \big) = \hat{\theta}_{MLE}
$$

<br>

책의 설명을 인용하자면 아래와 같다.  
 > One way to interpret maximum likelihood estimation is to view it as minimizing the dissimilarity between **the empirical distribution $\hat p_{data}$** defined by the training set and **the model distribution $p_{model}$**, with the degree of dissimilarity between the two measured by the KL divergence.  












