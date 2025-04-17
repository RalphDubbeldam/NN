import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

tfb = tfp.bijectors
tfd = tfp.distributions

# --- Define LeakyReLU Bijector ---
class LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.2, validate_args=False, name="leaky_relu"):
        super(LeakyReLU, self).__init__(forward_min_event_ndims=1, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, y / self.alpha)

    def _inverse_log_det_jacobian(self, y):
        event_dims = self._event_dims_tensor(y)
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, I / self.alpha)
        log_abs_det_J_inv = tf.math.log(J_inv)
        return tf.reduce_sum(log_abs_det_J_inv, axis=event_dims)

# --- Create the bijector ---
leaky_relu = LeakyReLU(alpha=0.2)

# --- 1D Visualization of LeakyReLU ---
x_1d = tf.linspace(-5.0, 5.0, 400)
y_1d = leaky_relu.forward(x_1d)

plt.figure(figsize=(6, 4))
plt.plot(x_1d, y_1d, label='LeakyReLU', color='blue')
plt.plot(x_1d, x_1d, '--', label='Identity (x)', color='gray')
plt.title("1D LeakyReLU Transformation")
plt.xlabel("Input x")
plt.ylabel("Output y = LeakyReLU(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 2D Transformation Visualization ---
# Base distribution: standard 2D Gaussian
base_dist = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
samples = base_dist.sample(1000)

# Transform samples using the LeakyReLU bijector
transformed_samples = leaky_relu.forward(samples)

# Convert to NumPy for plotting
samples_np = samples.numpy()
transformed_np = transformed_samples.numpy()

# --- Plot before and after transformation ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, edgecolors='k')
axs[0].set_title("Original Samples (2D Gaussian)")
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
axs[0].axis('equal')
axs[0].grid(True)

axs[1].scatter(transformed_np[:, 0], transformed_np[:, 1], alpha=0.5, edgecolors='k')
axs[1].set_title("After LeakyReLU Bijector")
axs[1].set_xlabel("x1'")
axs[1].set_ylabel("x2'")
axs[1].axis('equal')
axs[1].grid(True)

plt.tight_layout()
plt.show()
