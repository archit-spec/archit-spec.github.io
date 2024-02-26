
Policy gradient methods work by computing an estimator of the policy gradient and plugging it into a stochastic gradient ascent algorithm. The most commonly used gradient estimator has the form.
$$
g^\tau = E^\tau \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) A_t \right]

$$
