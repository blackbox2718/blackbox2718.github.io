---
title: Divergence and Curl
date: 2020-08-05 20:53:30 -0700
math: true
categories: [math-blog, calculus]
tags: [vector-calculus]
---
When I first learned about this topic in Multivariable Calculus, I think
I did not do a decent job to understand how elegant and beautiful the
concept is. In this post, I want to take a second and more detailed look
into the math and intuition behind the language that defines Maxwell's
Equation - divergence and curl.

# Definition
*Divergence* is a quantity associated with a vector filed which measures
how much a flow is expanding/shrinking at a given point. More precisely,
it is the rate of change, with respect to time, of the density of the
fluid. Let's assume we have a vector field defined on $$\mathbb{R}^3$$,
 $$F = \left\langle P, Q, R \right\rangle$$. Then, the divergence of F,
also written as $$\text{div }\textbf{F}$$, is
$$\text{div } \textbf{F} = \nabla \cdot \textbf{F} = \frac{\partial P}{\partial
    x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$. *Curl*, on the other hand, describes the rotation of a vector field in
3-dimensional Euclidean space or in other words, the tendency of a fluid
to swirl around at a given point. While divergence is a scalar field,
curl itself is a vector field which the curl of **F** can be defined as
following 

$$\begin{aligned}
    \text{curl }\textbf{F} &=
    \begin{vmatrix} 
        \mathbf{i} & \mathbf{j} & \mathbf{k}\\
        \frac{\partial }{\partial x} & \frac{\partial }{\partial y} &
        \frac{\partial }{\partial z} \\
        P & Q & R
    \end{vmatrix} \\
                           &= \left( \frac{\partial R}{\partial y} -
                           \frac{\partial Q}{\partial z}  \right)\mathbf{i} +
                           \left( \frac{\partial P}{\partial z} - \frac{\partial
                           R}{\partial x}  \right)\mathbf{j} + \left(
                       \frac{\partial Q}{\partial x} - \frac{\partial
                   P}{\partial y} \right)\mathbf{k} \\
                           &= \nabla \times \textbf{F}
    \end{aligned}$$ 
    
**Note**: In both cases, $$\nabla$$ is treated
as it were a vector \"packed\" with partial derivatives operator
$$\nabla = \left\langle \frac{\partial }{\partial x} , \frac{\partial
        }{\partial y}, \frac{\partial }{\partial z} \right\rangle$$ In
short, div and curl can be simply and mathematically described as

$$\begin{aligned}
        \text{div }\textbf{F} &= \nabla \cdot \textbf{F} \\
        \text{curl }\textbf{F} &= \nabla \times \textbf{F}
    \end{aligned}$$ 
    
It can be deduced that both the divergence and curl
of a vector field are correlated to the Jacobian matrix which is

$$JF(x,y,z) = 
    \begin{bmatrix} 
        \frac{\partial P}{\partial x} & \frac{\partial P}{\partial y} &
        \frac{\partial P}{\partial z}\\
        \frac{\partial Q}{\partial x} & \frac{\partial Q}{\partial y} &
        \frac{\partial Q}{\partial z} \\
        \frac{\partial R}{\partial x} & \frac{\partial R}{\partial y} &
        \frac{\partial R}{\partial z} 
    \end{bmatrix}$$ 
    
Notice that the sum of all the diagonal entries in
$$JF$$ is $$\text{div }\textbf{F}$$. Moreover, from Linear Algebra, we know that a real
matrix A can be expressed as $$A = \frac{1}{2}\left( A - A^\top \right)$$
where $$A$$ is a skew-symmetric matrix. Apply this notion to $$JF$$, we obtain
the following 

$$JF = \frac{1}{2}
    \begin{bmatrix} 
        0 & \frac{\partial P}{\partial y} - \frac{\partial Q}{\partial x} &
        \frac{\partial P}{\partial z} - \frac{\partial R}{\partial x} \\[1em]
        \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} & 0 &
        \frac{\partial Q}{\partial z} - \frac{\partial R}{\partial y} \\[1em]
        \frac{\partial R}{\partial x} - \frac{\partial P}{\partial z} &
        \frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z} & 0
    \end{bmatrix}$$
    
Now, we can observe that all the non-zero components of this matrix and their difference are indeed $$\text{curl }\textbf{F}$$.

# Visualization
Let us jump straight in the visualization of divergence and curl. There
are 3 cases here:

-   Zero

-   Negative

-   Positive

Since it's easier to visualize the concept in 2-dimensional space, I
will plot these vector fields in 2D plane, and without a doubt, things
would get tricky and maybe even deceiving when you view them from 3D
perspective.

Observe that the rotation of the two vector field appears to be opposite
in direction. The first one is counter-clockwise while the second one is
clockwise. The sign here is determined the similar way a cross product
is defined (by using the right-hand rule). Here, it is obvious that a
zero curl(irrotational) at a certain point means there is no rotation
occurring at that point in space.

![curl](/images/drafts/curl.png)

Looking at the vector field above would certainly give us a sense of
what does it mean to be a sink/source. In a rather analogous context,
one can imagine the sign of div at a point in space to be positive when
the fluid around that point tend to diverge away from it, and the sign
would be negative when the fluid converges toward that point -- like a
series converges to a certain value. A zero divergence indicates that
the net flux is zero(incompressible), i.e. the amount of fluid expanding
is exactly equal to the amount of contracting.

![div](/images/drafts/div.png)

There is a caveat though. Oftentimes, these pictures can grant us some
intuition about the behavior of a vector field, they, however, can be
quite misleading as well. One must be really careful when using these
vector fields to determine the sign of div/curl. The better alternative
should be performing a thorough calculation.

# Applications
Divergence and curl are used widely in math and physics, some of them
are

- Maxwell's Equation

- Stokes' Theorem

- Divergence Theorem

- Predator - Prey Model
