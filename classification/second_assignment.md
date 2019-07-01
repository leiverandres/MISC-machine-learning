# Assignment #2

**Leiver Andres Campeón,  Juan Sebastián Vega Patiño**

------



## 1. Maximum likelihood for prior probabilities

We want to maximize the log likelihood subject to the constraint that $\sum_{k=1}^{K} \pi_k = 1$. Introducing Lagrange multipliers:
$$
P(\pi, \Phi|\pi_k, \mu_k, \Sigma) = \sum_{n=1}^{N} \left[ \sum_{j=1}^{K}\left( t_{nj}(\ln \pi_j + \ln \mathcal{N}(\phi(X_n)|\mu_j, \Sigma)) \right) + \lambda \sum_{j=1}^{K}\pi_k - 1 \right]
$$
Lets calculate derivative respect to  $\pi_k $  and make it equal to $0$
$$
\frac{\partial P}{\partial \pi_k} = \frac{1}{\pi_k} \sum_{n=1}^{N}t_{nk} + \lambda = 0
$$

$$
-\lambda \pi_k = \sum_{n=1}^{N}t_{nk}
$$

$$
\pi_k = -\frac{1}{\lambda}\sum_{n=1}^{N}t_{nk} = -\frac{N_k}{\lambda}
$$

Where $N_k$ is the number of examples whose label belongs to class $k$. Then we're going to take the derivative respect to $\lambda$
$$
\frac{\partial P}{\partial \lambda} = \sum_{k=1}^{K}\pi_k - 1 = 0
$$

$$
 \sum_{k=1}^{K}\pi_k  = 1
$$

Now let's plug all the values of the $\pi_k$ for all $K$ classes and find $\lambda$
$$
\sum_{k=1}^{K} \pi_k = \sum_{k=1}^{K} -\frac{N_k}{\lambda} = -\frac{N}{\lambda} = 1
$$

$$
\lambda = -N
$$

After, once we have found the value for $\lambda$, we can plug it into the equation we resolved for $\pi_k$ obtaining the maximum likelihood that estimates the prior probabilities as:
$$
\pi_k = \frac{N_k}{N}
$$

## 2. Maximum likelihood for Gaussian distribution mean 

$$
P(\pi, \Phi | \pi_k, \mu_k, \Sigma) = \sum_{n=1}^{N} \bigg[ \sum_{j=1}^{K} \bigg(t_{nj} \big(\ln{\pi_j}
    + \ln{\mathcal{N}(\phi(X_n) | \mu_j, \Sigma)}\big) \bigg) \bigg]
$$

Let's derivate respect to $\mu_k$
$$
\frac{\partial P}{\partial \mu_k} = \sum_{n=1}^{N} \bigg[ t_{nk} \frac{\partial }{\partial \mu_k}  \big(\ln{\mathcal{N}(\phi(X_n) | \mu_k, \Sigma)}\big) \bigg] = 0
$$

$$
\sum_{n=1}^{N} \bigg[ t_{nk} \frac{\partial }{\partial \mu_k}  \big(\frac{-1}{2}(\phi_n - \mu_k)^T\Sigma^{-1}(\phi_n - \mu_k)\big) \bigg] = 0
$$

$$
\sum_{n=1}^{N} \bigg[ t_{nk} \big(\frac{-1}{2}-2\Sigma^{-1}(\phi_n - \mu_k)\big) \bigg] = 0
$$

$$
\Sigma^{-1} \sum_{n=1}^{N} \bigg[ t_{nk}(\phi_n - \mu_k) \bigg] = 0
$$

$$
\sum_{n=1}^{N} t_{nk}\phi_n - \sum_{n=1}^{N} t_{nk}\mu_k = 0
$$

$$
N_k\mu_k = \sum_{n=1}^{N} t_{nk}\phi_n
$$

$$
\mu_k = \frac{1}{N_k}\sum_{n=1}^{N} t_{nk}\phi_n
$$



## 3. Linear functions of $\phi$ components

We can describe a classification model for $K$ classes with all the characteristics pointed in the statement as:
$$
P(\Phi | C_k) = \prod_{m=1}^{M} \prod_{l=1}^{L} \mu_{kml}^{\phi_ml}
$$
Since:
$$ { }
a_k = \ln(P(\Phi | C_k)P(C_k))
$$
Then plugging first equation:
$$
a_k = \ln(\prod_{m=1}^{M} \prod_{l=1}^{L} \mu_{kml}^{\phi_ml}) + \ln(P(C_k))
$$

$$
a_k = \sum_{m=1}^{M} \sum_{l=1}^{L}  \phi_{ml} \mu_{kml} + \ln(P(C_k)
$$



## 4. Derivative of the softmax activation function

$$
p(C_{k}|\phi) = y_k(\phi) = \frac{\exp(a_k)}{\sum_j \exp(a_j)}
$$

We want to find the derivative of softmax respect to any activation
$$
\frac{\partial y_k}{\partial a_i} = \frac{\partial \frac{\exp(a_k)}{\sum_j \exp(a_j)}}{\partial a_i}
$$

For the quotient rule we got
$$
f(x) = \frac{g(x)}{h(h)}
$$

$$
f'(x) = \frac{g'(x) h(x) - h'(x) g(x)}{h(x)^2}
$$
So, by applying quotient rule to softmax, we have two cases

When $i = k$
$$
\frac{\partial y_k}{\partial a_i} = \frac{\exp(a_k) \sum_j \exp(a_j) - \exp(a_i) \exp(a_k)}{(\sum_j \exp(a_j))^2}
$$

$$
= \frac{ \exp(a_k) (\sum_j \exp(a_j) - \exp(a_i)) }{ (\sum_j \exp(a_j))^2 }
$$

$$
= \frac{\exp(a_k)}{\sum_j \exp(a_j)} \frac{\sum_j \exp(a_j) - \exp(a_i)}{\sum_j \exp(a_j)}
$$

Note that
$$
\frac{\exp(a_k)}{\sum_j \exp(a_j)} = y_k
$$
And
$$
\frac{\exp(a_i)}{\sum_j \exp(a_j)} = y_i
$$

$$
= y_k (1 - y_i)
$$

When $i \neq k$
$$
\frac{\partial y_k}{\partial a_i} = \frac{-\exp(a_i)\exp(a_k)}{(\sum_j \exp(a_j))^2}
$$

$$
= \frac{-\exp(a_i)}{\sum_j \exp(a_j)} \frac{\exp(a_k)}{\sum_j \exp(a_j)}
$$

$$
\frac{\partial y_k}{\partial a_i} = -y_i y_k
$$



## 5. Derivative of the cross entropy loss function

$$
E(w_{1}, ... ,w_{k}) = -\ln p(T|w_{1}, ...,w_{k}) = -\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\ln y_{nk}
$$

$$
\nabla_{w_{j}}E(w_1, ..., w_k) = -\sum_{n=1}^{N}\sum_{k=1}^{K} t_{nk} \frac{\partial \ln (y_{nk})}{\partial w_j}
$$

$$
= -\sum_{n=1}^{N}\sum_{k=1}^{K} t_{nk} \frac{1}{y_{nk}} \frac{\partial y_{nk}}{\partial z_{nk}} \frac{\partial z_{nk}}{\partial w_j}
$$



As we said in previous point:
$$
z_{nk} = W^{'}_k \phi_{n}
$$
Then it's derivative respect parameters would be
$$
\frac{\partial z_{nk}}{\partial w_j} = \phi_{n}
$$
Also, we calculated the derivative of softmax function and we found that it was defined for two cases.

Plugging both into our cross entropy equation we obtain:
$$
= -\sum_{n=1}^{N} \left[ \frac{t_{nj}}{y_{nj}} y_{nj} (1 - y_{nj}) + \sum_{k \neq j}^{K} \frac{t_{nk}}{y_{nk}} (-y_{nk}y_{nj}) \right] \phi_n
$$


​		- t_nj y_nj comes back into second sum
$$
-\sum_{n=1}^{N} \left[ t_{nj} - t_{nj}y_{nj} + \sum_{k \neq j}^{K} -t_{nk}y_{nj} \right] \phi_n
$$

$$
-\sum_{n=1}^{N} \left[ t_{nj} + \sum_{k=1}^{K} -t_{nk}y_{nj} \right] \phi_n
$$

$$
-\sum_{n=1}^{N} \left[ t_{nj} + y_{nj} \sum_{k=1}^{K} -t_{nk} \right] \phi_n
$$

$$
\sum_{k=1}^{K} t_{nk} = 1
$$


$$
-\sum_{n=1}^{N} \left[ t_{nj} - y_{nj} \right] \phi_n
$$

$$
\sum_{n=1}^{N} \left[ y_{nj} - t_{nj}\right] \phi_n
$$

So finally we get that the derivative of cross entropy loss function is
$$
\nabla_{w_{j}}E(w_1, ..., w_k) = \sum_{n=1}^{N}(y_{nj} - t_{nj})\phi_n
$$
