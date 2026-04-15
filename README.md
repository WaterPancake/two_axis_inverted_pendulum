---
layout: default
title: Two Axis Cart Pole
permalink: /two_axis_inverted_pendulum/
---

# Two Axis Cart Pole
is an extension of the canonical Cart Pole by adding another dimension of traversal and control to the cart, and another degree of freedom to the inverted pendulum.

The model used is Built off the the [inverted pendulum from Brax](https://github.com/google/brax/blob/main/brax/envs/assets/inverted_pendulum.xml) created by Google.

## Environment
The inverted two-axis pendulum can be represented as the following:

$\textbf{x} = \left[ x, y, \theta_x, \theta_y, \dot{x}, \dot y, \dot{\theta_x}, \dot{\theta_y} \right]$


| Variable | Description | minVal | maxVal
| ----- | ----- | --- | --- |
| $x$ | carts position along x-axis | -2 | 2 |
| $y$ | carts position along y-axis | -2 | 2 |
| $\theta_x$ | poles angle along the x axis about the cart in radians | $0$ | $2\pi$ |
| $\theta_y$ | poles angle along the y axis about the cart in radian | $0$ | $2\pi$|
| $\dot x$ | be the linear velocity of cart along x-axis | $-\infty$ | $\infty$ |
| $\dot y$ | be then linear velocity of cart along y-axis | $-\infty$ | $\infty$ |
| $\dot \theta_x$ | the angular velocity of the pole along the x axis about the cart | $-\infty$ | $\infty$ |
| $\dot \theta_y$ | the angular velocity of the pole along the y axis about the car | $-\infty$ | $\infty$ |

The system has control inputs:
$$u = \left[ F_x, F_y \right]^T$$

corresponding to linear force applied to the cart in the $x$ and $y$ axis respectfully.


