---
layout: post
title: "Collapsed Gibbs Sampler for LDA"
tags: [Statistics]
comments: true
---

이 포스트는 [**"Griffiths, Steyvers - Finding scientific topics, 2004"**](https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf?__=)의 내용을 발췌독 및 정리한 것이다. 이 paper는 collapsed Gibbs sampler를 이용해 LDA의 inference를 수행하는 방법을 제시한다. 

## Introduction

앞에서는 $p(\mathbf w \vert \alpha, \beta)$의 계산은 다루기 힘들기 때문에 variational Bayes 방법을 사용하여 model hyperparameter를 추정하였다. 여기서는 다소 다른 접근법을 소개한다. Collapsed Gibbs sampler를 이용한 LDA는 앞의 방법들처럼 $\beta, \theta$를 추정해야 할 parameter로 보지 않고, multinomial-Dirichlet conjugacy를 이용하여 integrate out한다. 그 다음, 관측된 document $\mathbf w$에 대한 topic variable $\mathbf z$의 posterior distribution $p(\mathbf z \vert \mathbf w)$를 Gibbs sampling을 통해 추정한다. $\beta, \theta$에 대한 추정치는 이 posterior로부터 얻을 수 있다.

그를 위해 topic이 주어졌을 때 word probability를 나타내는 $\beta$를 model hyperparameter로 보지 않고, prior를 부여하자. 그 때의 complete probability model은 다음과 같다. 붉은 글씨로 적힌 $\boldsymbol \beta$가 원래 LDA에서는 model  parameter였으며, 여기서는 Dirichlet prior를 새로 부여한 부분이다.

$$
\begin{align*}
\boldsymbol \theta_d &\sim \text{Dirichlet}(\alpha) \quad \quad \quad d=1,\cdots, D\\
z_{dn} \vert \boldsymbol \theta_d &\sim \text{Multinomial}(\boldsymbol \theta) \quad \quad  d=1,\cdots, D,\enspace  n= 1,\cdots, N_d \quad \\
\textcolor{red}{\boldsymbol \beta_i} &\text{ }\textcolor{red}{\sim \text{Dirichlet}(\eta) \quad  \quad \quad i= 1, \cdots, k}\\
w_{dn} \vert z_{dn}, \boldsymbol \beta &\sim \text{Multinomial}(\boldsymbol \beta_{z_{dn}}) \quad  d=1,\cdots, D,\enspace  n= 1,\cdots, N_d \quad 
\end{align*}
$$

이제 우리의 hyperparameter는 $\alpha, \eta$가 주어졌을 때, topic assignment $z_{dn}$의 conditional을 구하자.



## Deriving conditionals

어떤 한 document 내에서, $d$번째 document의 $n$번째 word에 대한 topic assignment $z_{dn}$의 conditional은 다음과 같다.

$$
\begin{align*}
p(z_{dn} \vert \mathbf Z_{-dn}, \mathbf W, \alpha, \eta) &= \frac{p(\mathbf Z , \mathbf W \vert \alpha, \eta)}{p( \mathbf Z_{-dn}, \mathbf W \vert  \alpha, \eta)} \\
&=\frac{p(\mathbf Z , \mathbf W \vert \alpha, \eta)}{p( \mathbf Z_{-dn}, \mathbf W_{-dn} \vert  \alpha, \eta)} p( w_{dn} \vert \mathbf Z_{-dn}, \mathbf W_{-dn} , \alpha, \eta) \\
&\propto  \frac{p(\mathbf Z , \mathbf W \vert \alpha, \eta)}{p( \mathbf Z_{-dn}, \mathbf W_{-dn} \vert  \alpha, \eta)} \\
 ( \because \enspace p( w_{dn} \vert \mathbf Z_{-dn}, \mathbf W_{-dn} &, \alpha, \eta) \text{ is a constant with respect to }z_{dn} )
\end{align*}
$$


분자인 $p(\mathbf Z , \mathbf W \vert \alpha, \eta)$을 전개해보면 다음과 같다.

$$
\begin{align*}
p(\mathbf Z , \mathbf W \vert \alpha, \eta) 
&= \int_{\boldsymbol\theta^{(1:D)}} \int_{\boldsymbol\beta_{1:k}} p(\mathbf Z , \mathbf W , \boldsymbol\beta_{1:k} , \boldsymbol\theta^{(1:D)} \vert \alpha, \eta) d\boldsymbol\beta_{1:k} d\boldsymbol\theta^{(1:D)} \\
&=\int_{\boldsymbol\beta_{1:k}} p(\mathbf W  \vert \mathbf Z ,\boldsymbol\beta_{1:k})p(\boldsymbol\beta_{1:k}  \vert \eta) 
d\boldsymbol\beta_{1:k} 
\int_{\boldsymbol\theta^{(1:D)}}  p(\mathbf Z  \vert \boldsymbol\theta^{(1:D)}) p( \boldsymbol\theta^{(1:D)} \vert \alpha)  d\boldsymbol\theta^{(1:D)} \\
\end{align*}
$$

두 적분식을 각각 정리하면 다음과 같다.

$$
\begin{align*}
1. \enspace \int_{\boldsymbol\beta_{1:k}} p(\mathbf W  \vert \mathbf Z ,\boldsymbol\beta_{1:k})p(\boldsymbol\beta_{1:k}  \vert \eta) 
d\boldsymbol\beta_{1:k} 
&= \int_{\boldsymbol\beta_{1:k}}
p(\mathbf W  \vert \mathbf Z ,\boldsymbol\beta_{1:k})\prod_{i=1}^{k} p(\boldsymbol\beta_{i}  \vert \eta)  d\boldsymbol\beta_{1:k}\\
&= \int_{\boldsymbol\beta_{1:k}}
\prod_{d=1}^{D} p(\mathbf w_{d}  \vert \mathbf z_{d} ,\boldsymbol\beta_{1:k})\prod_{i=1}^{k} p(\boldsymbol\beta_{i}  \vert \eta)  d\boldsymbol\beta_{1:k}\\
&= \int_{\boldsymbol\beta_{1:k}}
\left( \prod_{d=1}^{D}\prod_{i=1}^k \prod_{j=1}^V \beta_{ij}^{ \sum_{n=1}^{N_d} w_{dn}^j z_{dn}^i} \right)
\left( \prod_{i=1}^k \text{Dirichlet}( \beta_i ; \eta ) \right)
d\boldsymbol\beta_{1:k} \\
&= \int_{\boldsymbol\beta_{1:k}}
\left( \prod_{i=1}^k \prod_{j=1}^V \beta_{ij}^{\sum_{d=1}^{D} \sum_{n=1}^{N_d} w_{dn}^j z_{dn}^i} \right)
\prod_{i=1}^k \left( \frac{\Gamma(\sum_{j=1}^{V} \eta_j)}{\prod_{j=1}^{V}\Gamma(\eta_j)} \prod_{j=1}^{V} \beta_{ij}^{\eta_j -1} \right)
d\boldsymbol\beta_{1:k} \\
\text{Let } \Xi_{i,j} = \sum_{d=1}^{D} \sum_{n=1}^{N_d} w_{dn}^j z_{dn}^i : \text{ counts}& \text{ of }j\text{th word in }i\text{th topic across all documents.}\\
&= \left[ \frac{\Gamma(\sum_{j=1}^{V} \eta_j)}{\prod_{j=1}^{V}\Gamma(\eta_j)} \right]^k
\int_{\boldsymbol\beta_{1:k}}
 \prod_{i=1}^k \prod_{j=1}^V \beta_{ij}^{(\eta_j +\Xi_{i,j}  )-1} 
d\boldsymbol\beta_{1:k} \\
&= \left[ \frac{\Gamma(\sum_{j=1}^{V} \eta_j)}{\prod_{j=1}^{V}\Gamma(\eta_j)} \right]^k
\prod_{i=1}^k \frac{\prod_{j=1}^{V}\Gamma(\eta_j +\Xi_{i,j} )}{\Gamma(\sum_{j=1}^{V} \eta_j + \Xi_{i,\bullet} )} \\
&= \prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{i})}{\text{Beta}(\eta )} \\
&(\text{where }\text{ } \Xi_i = [\Xi_{i,1},\cdots, \Xi_{i,V}]^T)
\end{align*}
$$

$$
\begin{align*}
2. \enspace \int_{\boldsymbol\theta^{(1:D)}}  p(\mathbf Z  \vert \boldsymbol\theta^{(1:D)}) p( \boldsymbol\theta^{(1:D)} \vert \alpha)  d\boldsymbol\theta^{(1:D)}
&= \int_{\boldsymbol\theta^{(1:D)}}   
\prod_{d=1}^{D}p(\mathbf z_d \vert \boldsymbol\theta^{(d)}) p( \boldsymbol\theta^{(d)} \vert \alpha)  d\boldsymbol\theta^{(1:D)}\\
&= \int_{\boldsymbol\theta^{(1:D)}}   
\left( \prod_{d=1}^{D} \prod_{n=1}^{N_d} \prod_{i=1}^{k} \theta_{di}^{z_{dn}^{i}} \right)
\left( \prod_{d=1}^{D} \text{Dirichlet}(\boldsymbol \theta_d ; \alpha) \right)
d\boldsymbol\theta^{(1:D)} \\
&= \int_{\boldsymbol\theta^{(1:D)}}   
\left( \prod_{d=1}^{D} \prod_{i=1}^{k} \theta_{di}^{\sum_{n=1}^{N_d}z_{dn}^{i}} \right)
\prod_{d=1}^{D} \left( \frac{\Gamma(\sum_{i=1}^{k} \alpha_i)}{\prod_{i=1}^{k}\Gamma(\alpha_i)} \prod_{i=1}^{k} \theta_{di}^{\alpha_i-1} \right)
d\boldsymbol\theta^{(1:D)} \\
&= 
\left[ \frac{\Gamma(\sum_{i=1}^{k} \alpha_i)}{\prod_{i=1}^{k}\Gamma(\alpha_i)} \right]^D
\int_{\boldsymbol\theta^{(1:D)}}   
\left( \prod_{d=1}^{D} \prod_{i=1}^{k} \theta_{di}^{(\alpha_i +  \sum_{n=1}^{N_d}z_{{d^\prime}n}^{i}) -1} \right)
d\boldsymbol\theta^{(1:D)} \\
\text{Let } \Omega_{d,i} = \sum_{n=1}^{N_d}z_{dn}^{i} : \text{ the number} \text{ of } &\text{words } \text{of }i\text{th topic in }d\text{th document.}\\
&= \left[ \frac{\Gamma(\sum_{i=1}^{k} \alpha_i)}{\prod_{i=1}^{k}\Gamma(\alpha_i)} \right]^D
\int_{\boldsymbol\theta^{(1:D)}}   
\left( \prod_{d=1}^{D} \prod_{i=1}^{k} \theta_{di}^{(\alpha_i +  \Omega_{d,i}) -1} \right)
d\boldsymbol\theta^{(1:D)} \\
&= \left[ \frac{\Gamma(\sum_{i=1}^{k} \alpha_i)}{\prod_{i=1}^{k}\Gamma(\alpha_i)} \right]^D
\prod_{d=1}^D \frac{\prod_{i=1}^{k} \Gamma(\alpha_i +  \Omega_{d,i})}{\Gamma(\sum_{i=1}^{k}\alpha_i +  \Omega_{d,\bullet})} \\
&= \prod_{d=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_d)}{\text{Beta}(\alpha )} \\
&(\text{where }\text{ } \Omega_d = [\Omega_{d,1},\cdots, \Omega_{d,k}]^T)
\end{align*}
$$

따라서 위 식의 분자 $p(\mathbf Z , \mathbf W \vert \alpha, \eta)$는 다음과 같다.

$$
p(\mathbf Z , \mathbf W \vert \alpha, \eta) = \prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{i})}{\text{Beta}(\eta )}\prod_{{d^\prime}=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_{{d^\prime}})}{\text{Beta}(\alpha )}
$$

또한 같은 방법으로 분모인 $ p (\mathbf Z_{-dn}, \mathbf W_{-dn} \vert \alpha, \eta)$도 구할 수 있다.

$$
p(\mathbf Z_{-dn} , \mathbf W_{-dn} \vert \alpha, \eta) = \prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{i}^{-dn})}{\text{Beta}(\eta )}\prod_{{d^\prime}=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_{{d^\prime}}^{-dn})}{\text{Beta}(\alpha )}
$$

$\Xi_{i,j}^{-dn},  \Omega_{d,i}^{-dn}$은 $d$번째 document의 $n$번째 word와 topic variable을 제외하고 구한 $\Xi_{i,j},  \Omega_{d,i}$이다. 이제 우리는 다음과 같이 conditional을 적을 수 있다.

$$
\begin{align*}
p(z_{dn} \vert \mathbf Z_{-dn}, \mathbf W, \alpha, \eta) &\propto  \frac{p(\mathbf Z , \mathbf W \vert \alpha, \eta)}{p( \mathbf Z_{-dn}, \mathbf W_{-dn} \vert  \alpha, \eta)} \\
&=\frac{\prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{i})}{\text{Beta}(\eta )}\prod_{{d^\prime}=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_{d^\prime})}{\text{Beta}(\alpha )} }
{
\prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{i}^{-dn})}{\text{Beta}(\eta )}\prod_{{d^\prime}=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_{{d^\prime}}^{-dn})}{\text{Beta}(\alpha )}} \\
&=\prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{i})}{\text{Beta}(\eta +\Xi_{i}^{-dn})}\prod_{d^\prime=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_{d^\prime})}{\text{Beta}(\alpha +  \Omega_{d^\prime}^{-dn})} 
\end{align*}
$$


Conditional을 구하고자 하는 $\textcolor{green}{\tilde d}$번째 document의 $n$번째 word가 $\textcolor{blue}{\tilde j}$이고, 이 word는 topic $\textcolor{red}{\tilde i}$에서 생성되었다고 하자. $\Xi_{i,j}^{-dn},  \Omega_{d,i}^{-dn}$은 한 word(observation)을 제외하고 구한 $\Xi_{i,j},  \Omega_{d,i}$이므로, 다음이 만족한다.

$$
\text{If }i=\textcolor{red}{\tilde i}, j=\textcolor{blue}{\tilde j}, \quad \Xi_{i,j}^{-\textcolor{green}{\tilde d} n} = \Xi_{i,j} -1 . \quad \text{else same.}\\
\text{If }d= \textcolor{green}{\tilde d},i=\textcolor{red}{\tilde i}, \quad \Omega_{d,i}^{-\textcolor{green}{\tilde d}n} = \Omega_{d,i}-1. \quad \text{else same.}
$$


이제 첫 번째 multiplicant를 구하자.

$$
\begin{align*}
\prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{\textcolor{red}{\tilde i}})}{\text{Beta}(\eta +\Xi_{\textcolor{red}{\tilde i}}^{-\textcolor{green}{\tilde d}n})} 
&= \frac{\text{Beta}(\eta +\Xi_{\textcolor{red}{\tilde i}})}{\text{Beta}(\eta +\Xi_{\textcolor{red}{\tilde i}}^{-\textcolor{green}{\tilde d}n})}  \\
&=\frac{\Gamma(\sum_{j=1}^{V} \eta_j + \Xi_{\textcolor{red}{\tilde i},\bullet}^{-\textcolor{green}{\tilde d}n} )}{\Gamma(\sum_{j=1}^{V} \eta_j + \Xi_{\textcolor{red}{\tilde i},\bullet} )} \frac{\prod_{j=1}^{V}\Gamma(\eta_j +\Xi_{\textcolor{red}{\tilde i},j} )}{\prod_{j=1}^{V}\Gamma(\eta_j +\Xi_{\textcolor{red}{\tilde i},j}^{-\textcolor{green}{\tilde d}n} )} \\
&=\frac{1}{\sum_{j=1}^{V} \eta_j + \Xi_{\textcolor{red}{\tilde i},\bullet}^{-\textcolor{green}{\tilde d}n} } \frac{\eta_\textcolor{blue}{\tilde j} +\Xi_{\textcolor{red}{\tilde i},\textcolor{blue}{\tilde j}}^{-\textcolor{green}{\tilde d}n} }{1} \\
&=\frac{\eta_\textcolor{blue}{\tilde j} +\Xi_{\textcolor{red}{\tilde i},\textcolor{blue}{\tilde j}}^{-\textcolor{green}{\tilde d}n} }{\sum_{j=1}^{V} (\eta_j + \Xi_{\textcolor{red}{\tilde i},j}^{-\textcolor{green}{\tilde d}n} )} 
\end{align*}
$$

두 번째 multiplicant는 다음과 같이 구한다.

$$
\begin{align*}
\prod_{d=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_{d})}{\text{Beta}(\alpha +  \Omega_{d}^{-\textcolor{green}{\tilde d} n})}  &= \frac{\text{Beta}(\alpha +  \Omega_{\textcolor{green}{\tilde d}})}{\text{Beta}(\alpha +  \Omega_{\textcolor{green}{\tilde d}}^{-\textcolor{green}{\tilde d} n})} \\
&= \frac{\Gamma(\sum_{i=1}^{k}\alpha_i +  \Omega_{\textcolor{green}{\tilde d},\bullet}^{-\textcolor{green}{\tilde d}n})}{\Gamma(\sum_{i=1}^{k}\alpha_i +  \Omega_{\textcolor{green}{\tilde d},\bullet})} \frac{\prod_{i=1}^{k} \Gamma(\alpha_i +  \Omega_{\textcolor{green}{\tilde d},i})}{\prod_{i=1}^{k} \Gamma(\alpha_i +  \Omega_{\textcolor{green}{\tilde d},i}^{-\textcolor{green}{\tilde d}n})} \\
&= \frac{1}{\sum_{i=1}^{k}\alpha_i +  \Omega_{\textcolor{green}{\tilde d},\bullet}^{-\textcolor{green}{\tilde d}n}} \frac{\alpha_\textcolor{red}{\tilde i} +  \Omega_{\textcolor{green}{\tilde d},\textcolor{red}{\tilde i}}^{-\textcolor{green}{\tilde d}n}}{1} \\
&= \frac{\alpha_\textcolor{red}{\tilde i} +  \Omega_{\textcolor{green}{\tilde d},\textcolor{red}{\tilde i}}^{-\textcolor{green}{\tilde d}n}}{\sum_{i=1}^{k}(\alpha_i +  \Omega_{\textcolor{green}{\tilde d},i}^{-\textcolor{green}{\tilde d}n})}
\end{align*}
$$


따라서 도출한 conditional은 다음과 같다.

$$
\begin{align*}
p(z_{\textcolor{green}{\tilde d}n}=\textcolor{red}{\tilde i} \vert \mathbf Z_{-\textcolor{green}{\tilde d}n}, \mathbf W, \alpha, \eta) &\propto  \frac{p(\mathbf Z , \mathbf W \vert \alpha, \eta)}{p( \mathbf Z_{-\textcolor{green}{\tilde d}n}, \mathbf W_{-\textcolor{green}{\tilde d}n} \vert  \alpha, \eta)} \\

&=\prod_{i=1}^{k} \frac{\text{Beta}(\eta +\Xi_{i})}{\text{Beta}(\eta +\Xi_{i}^{-\textcolor{green}{\tilde d}n})}\prod_{d=1}^{D} \frac{\text{Beta}(\alpha +  \Omega_{d})}{\text{Beta}(\alpha +  \Omega_{d}^{-\textcolor{green}{\tilde d}n})} \\
&=
\frac{\eta_\textcolor{blue}{\tilde j} +\Xi_{\textcolor{red}{\tilde i},\textcolor{blue}{\tilde j}}^{-\textcolor{green}{\tilde d}n} }{\sum_{j=1}^{V} (\eta_j + \Xi_{\textcolor{red}{\tilde i},j}^{-\textcolor{green}{\tilde d}n} )} 
\frac{\alpha_\textcolor{red}{\tilde i} +  \Omega_{\textcolor{green}{\tilde d},\textcolor{red}{\tilde i}}^{-\textcolor{green}{\tilde d}n}}{\sum_{i=1}^{k}(\alpha_i +  \Omega_{\textcolor{green}{\tilde d},i}^{-\textcolor{green}{\tilde d}n})}
\end{align*}
$$


## Algorithm

도출한 conditional을 가지고 Gibbs sampling을 수행하는 것은 어렵지 않다. 눈여겨볼 부분은 conditional 식을 알고리즘에 적용하는데 필요한 것은 $\Xi_{i,j}^{-dn},  \Omega_{d,i}^{-dn}$, 즉 **'$i$ 번째 topic에 word $j$가 지정된 횟수'**와 '**$d$ 번째 document 내에서 $i$ 번째 topic으로부터 생성된 단어의 수**'뿐이다. Collapsed Gibbs sampling을 이용한 LDA의 수행 과정은 다음과 같다.



**1. Initialization**

* Count variable, $\Xi_{i,j}, \Xi_{i,\bullet},  \Omega_{d,i}, \Omega_{d,\bullet}$을 $0$으로 설정.

* $\mathtt{for}$ all documents $d = 1, \cdots , D$, $\mathtt{do}$

  * $\mathtt{for}$ all words $n = 1, \cdots , N_d$, $\mathtt{do}$
    * Sample topic variable $z_{dn} \sim \text{Multinomial}(\frac{1}{k}, \cdots , \frac{1}{k})$.
    * Increment : $\Xi_{z_{dn},j} = \Xi_{z_{dn},j} + 1$.
    * Increment : $\Xi_{z_{dn},\bullet}=\Xi_{z_{dn},\bullet}+1$.
    * Increment : $\Omega_{d,z_{dn}}=\Omega_{d,z_{dn}}+1$.
    * Increment : $\Omega_{d,\bullet}=\Omega_{d,\bullet}+1.$
  * $\mathtt{end} \text{ }\mathtt{for}$

* $\mathtt{end} \text{ }\mathtt{for}$


**2. Run Gibbs sampling**

* $\mathtt{while}$ not converged, $\mathtt{do}$
  * $\mathtt{for}$ all documents $d = 1, \cdots , D$, $\mathtt{do}$
    - $\mathtt{for}$ all words $n = 1, \cdots , N_d$, $\mathtt{do}$
      - Current topic assignment of $k$ for word $w_{dn}=j$,
      - Decrement : $\Xi_{k,j} = \Xi_{k,j} - 1$.
      - Decrement : $\Xi_{k,\bullet} = \Xi_{k,\bullet} - 1$.
      - Decrement : $\Omega_{d,k}=\Omega_{d,k}-1$.
      - Decrement : $\Omega_{d,\bullet}=\Omega_{d,\bullet}-1$.
      - Sample $\tilde k = z_{dn} \sim p(z_{dn} \vert \mathbf Z_{-dn}, \mathbf W, \alpha, \eta) $ with decremented $\Xi_{i,j}, \Xi_{i,\bullet},  \Omega_{d,i}, \Omega_{d,\bullet}$.
      - Increment : $\Xi_{\tilde k,j} = \Xi_{\tilde k,j} + 1$.
      - Increment : $\Xi_{\tilde k,\bullet} = \Xi_{\tilde k,\bullet} + 1$.
      - Increment : $\Omega_{d,\tilde k}=\Omega_{d,\tilde k}+1$.
      - Increment : $\Omega_{d,\bullet}=\Omega_{d,\bullet}+1$.
    - $\mathtt{end} \text{ }\mathtt{for}$
  * $\mathtt{end} \text{ }\mathtt{for}$
* $\mathtt{end} \text{ }\mathtt{while}$
