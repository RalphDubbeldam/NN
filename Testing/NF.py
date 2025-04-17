import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Disable eager execution
tf.compat.v1.disable_eager_execution()

tfd = tfp.distributions
tfb = tfp.bijectors

# Custom LeakyReLU bijector
class LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="leaky_relu"):
        super(LeakyReLU, self).__init__(forward_min_event_ndims=1, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        event_dims = tf.shape(y)[-1]
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        log_abs_det_J_inv = tf.math.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=-1)

batch_size = 512
x2_dist = tfd.Normal(loc=0., scale=4.)
x2_samples = x2_dist.sample(batch_size)
x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                scale=tf.ones(batch_size, dtype=tf.float32))
x1_samples = x1.sample()
x_samples = tf.stack([x1_samples, x2_samples], axis=1)
base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], tf.float32))

d, r = 2, 2
DTYPE = tf.float32
bijectors = []
num_layers = 6

class CustomAffine(tfb.Bijector):
    def __init__(self, scale, shift, validate_args=False, name="custom_affine"):
        super().__init__(forward_min_event_ndims=1, validate_args=validate_args, name=name)
        self.scale = scale
        self.shift = shift

    def _forward(self, x):
        return tf.matmul(x, self.scale) + self.shift

    def _inverse(self, y):
        # Reshape self.shift to be broadcastable with y
        shift_broadcasted = tf.reshape(self.shift, [1, -1])  # Shape will be (1, 2)
        y_shifted = y - shift_broadcasted
        return tf.linalg.matmul(y_shifted, tf.linalg.inv(self.scale))

    def _inverse_log_det_jacobian(self, y):
        # Compute the determinant of the scale matrix
        det = tf.linalg.det(self.scale)
        
        # Add a small epsilon to avoid zero determinant
        epsilon = 1e-6
        det = tf.maximum(det, epsilon)  # Prevent det from becoming zero
        
        # Log of the absolute determinant
        log_det = tf.math.log(tf.abs(det))
        
        # Return the log determinant as is (it's a scalar)
        return log_det

# Ensure V, shift, L, alpha are tf.Variable
for i in range(num_layers):
    V = tf.Variable(tf.random.normal([d, r], dtype=DTYPE), name=f'V_{i}', trainable=True)  # factor loading
    shift = tf.Variable(tf.random.normal([d], dtype=DTYPE), name=f'shift_{i}', trainable=True)  # affine shift
    L = tf.Variable(tf.random.normal([d * (d + 1) // 2], dtype=DTYPE), name=f'L_{i}', trainable=True)  # lower triangular
    bijectors.append(CustomAffine(
        scale=tf.linalg.matmul(V, V, transpose_b=True),  # Affine scaling
        shift=shift,
    ))
    alpha = tf.Variable(tf.abs(tf.random.normal([], dtype=DTYPE))) + .01
    bijectors.append(LeakyReLU(alpha=alpha))

# Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
mlp_bijector = tfb.Chain(
    list(reversed(bijectors[:-1])), name='mlp_bijector_2d')  # Changed name to a valid identifier
dist = tfd.TransformedDistribution(
    distribution=base_dist,
    bijector=mlp_bijector
)

# Set up the optimizer
optimizer = tf.optimizers.Adam(1e-3)

# Loss function and training step
def train_step():
    with tf.GradientTape() as tape:
        loss_value = dist.log_prob(x_samples)
        np_loss = -tf.reduce_mean(loss_value)
    
    # Ensure that grads are being calculated on tf.Variable objects
    grads = tape.gradient(np_loss, [V, shift, L, alpha])  # Variables that require gradients
    
    # Apply gradients only to tf.Variable objects
    optimizer.apply_gradients(zip(grads, [V, shift, L, alpha]))
    return np_loss

# Training loop
NUM_STEPS = int(1e5)
global_step = []
np_losses = []

for i in range(NUM_STEPS):
    np_loss = train_step()

    if i % 1000 == 0:
        global_step.append(i)
        np_losses.append(np_loss.numpy())
    if i % int(1e4) == 0:
        print(i, np_loss.numpy())

# Optionally plot the loss values
plt.plot(global_step, np_losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
