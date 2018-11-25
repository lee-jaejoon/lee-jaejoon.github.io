---
layout: post
title: "Linear Regression in R"
tags: [R]
comments: true
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Loading Packages and Dataset

```{r}
if(!require(ggplot2)){install.packages('ggplot2')}
library(ggplot2)
```


`ggplot2` 패키지에 내장된 `diamonds` 데이터를 사용하여 회귀분석을 해보려고 한다. `diamonds`는 10개의 변수, 53940개의 자료를 가진 데이터셋이다. 여러 변수가 있지만 그 중에서도 `price`변수, 즉 다이아몬드의 가격을 반응변수로 두고, 다이아몬드의 다른 특징들을 나타내는 `carat`,`cut`,`color`,`clarity` 등을 이에 대한 설명변수로 사용하자. 그 중에서도 `cut`, `color`, `clarity`는 숫자형 변수가 아닌 범주형 변수인데, 범주형 설명변수는 회귀분석에서 dummy variable을 이용하여 반영하게 된다.

```{r}
str(diamonds)
```

## Interpreting R's Regression Output
R에서 회귀분석을 수행해주는 함수는 `lm()`함수이다. `lm()`함수는 변수관계를 나타내는 formula옵션과, 변수들이 소속한 데이터를 나타내주는 data옵션을 지정해주어야 한다.  formula는 `반응변수 ~ 설명변수1 + 설명변수2 + 설명변수3`의 형식으로 작성하면 된다. 만약 반응변수로 지정한 변수 외의 data 내 모든 변수들을 설명변수로 넣고 싶다면, `반응변수 ~ .`의 형식으로 작성하면 된다.  

```{r}
fit <- lm(price ~. , data=diamonds)
summary(fit)
```
<br>
얻어진 회귀분석 결과를 어떻게 읽어야 할 지 하나하나 살펴보자.  
<br>
`Residuals:`는 실제 반응변수 값과 fitted value 사이의 차이인 잔차의 분포에 대한 정보를 알려준다.  
`Coefficients:`는 각 설명변수의 회귀계수($\beta$)에 대한 추정치($\hat{\beta}$)의 값, 표준오차, t-test 검정통계량, t-test p-value를 알려주며, 아래의 `Signif. codes` 기준대로 p-value의 크기에 따라 별(*)의 개수로 통계적 유의성의 정도를 나타내준다.  
`Residual standard error:`는 estimated standard deviation of residuals와 같은 의미이며, 다시 말해서 오차항의 표준편차(${\sigma}$)에 대한 추정값인 Mean Squared Error이다. 해당 자유도는 자료의 개수-회귀계수의 개수(n-p)와 같다.  
`R-squared:`와 `adjusted R-squared:`는 말그대로 R-squared와 adjusted R-squared의 값을 알려주는 것이며, 마지막의 `F-statistic:`은 회귀모형에 대한 F-test의 결과를 알려준다.

## Interpreting R's Diagnostic Plots
R의 `lm()`함수로 생성한 결과를 `plot()`함수에 넣으면 모형 진단에 도움이 되는 plot 네 개를 그려준다. 네 개의 plot이 각각 어떻게 그려진 plot이고, 어떤 의미가 있는지 살펴보자.
```{r}
plot(fit)
```

###1.Residuals vs fitted
```{r}
reg.plot <- plot(fit, which=1)
```


가로축에 fitted value를, 세로축에 residual을 그린 그림이다. 우리는 세션자료에서 평균=0, 등분산, 상호독립의 오차항을 가정했다. 하지만 우리는 오차를 관찰할 수 없으므로, 잔차의 모양을 보고 오차에 대한 내용을 추측할 수 밖에 없다. 만약 오차항에 대한세 가정이 잘 만족한다면, Residual vs fitted 그림은 residual=0의 가로선을 기준으로 규칙성 없이 오르내리며, 너비가 일정한 horizontal band 모양을 띄게 될 것이다.

###2. Normal Q-Q
```{r}
reg.plot <- plot(fit, which=2)
```

우리는 평균=0, 등분산, 상호독립 외에도 오차항이 정규분포를 따른다는 가정을 하였다. Q-Q plot은 이론적인 분포와 실제 분포가 일치하는지 판단하기 위해 그리는 그림이며, 이론적인 분포와 실제 분포가 일치한다면 우상향 대각선을 따라 점들이 위치하게 된다. R의 `lm()`함수는 얻어진 residual 값들에 대해 표준화를 한 studentized residual과 표준정규분포를 비교하는 Q-Q plot을 제공한다.

###3. Scale-location plot
```{r}
reg.plot <- plot(fit, which=3)
```

이 plot은 가로축에 fitted value를, 세로축에 residual을 그렸던 첫 번째 plot과 유사하지만, 세로축에 표준화된 잔차, studentized residual의 제곱근을 그렸다는 차이점이 있다. Residual vs fitted plot과 마찬가지로 plot에 눈에 띄는 규칙성이나 패턴, 혹은 크게 벗어난 값이 없어야 한다. 점들의 추세를 나타내는 붉은 선 역시 수평선을 이루어야 한다.

###4. Residuals vs leverage
```{r}
reg.plot <- plot(fit, which=5)
```

이 plot은 다른 observation에 비해 영향력이 더 큰 observation을 찾기 위해 사용되는 plot이다. 여기서 특정 observation의 영향력이 크다(influential)고 표현한 것은 단순히 다른 점들보다 극단적인 값을 갖는 outlier라는 뜻이 아니라, 회귀계수 추정치의 값을 결정하는데 영향력이 크다는 것이다. 즉 다시 말해서, 해당 observation을 제외하고 회귀분석을 진행했을 때 회귀계수의 추정치 값이 많이 달라진다는 의미이다. Cook's distance는 이러한 아이디어를 기반으로 influential observation을 찾는데 도움을 주는 척도이다. 등고선으로 나타난 Cook's distance에서 크게 벗어난 점이 있다면 이 observation을 influential point로 의심해 볼 수 있다.
