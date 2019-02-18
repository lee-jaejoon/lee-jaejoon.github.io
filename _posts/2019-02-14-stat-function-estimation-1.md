---
layout: post
title: "Statistical Learning as Function Approximation: 1. Introduction and Regularization"
tags: [Statistics]
comments: true
---

이 글과 이어지는 두 개의 글에서는 함수추정의 관점에서 통계 학습을 바라보는 modern statistical learning theory에 대해 소개하려고 한다. 시리즈의 첫 글인 이 포스트에서 hypothesis space, loss function, empirical error, generalization error 등의 기본 개념에 대해 소개한 뒤, Empirical Risk Minimization(ERM) 접근과 그 필요조건에 대해 소개한다. 그리고 이 필요조건들을 만족시키기 위한 방법으로서 Tikhonov regularization에 대해 소개할 것이다.

