"""
Made with <3 by nalinimsingh and mbchang
"""

from collections import namedtuple
import os
import time

import matplotlib.pyplot as plt
import numpy as onp
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.config import config
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, FanInConcat, FanOut, Relu, Softplus
import datasets

GaussianParams = namedtuple('GaussianParams', ('mu', 'logvar'))
latent_dim = 64

encoder_init, encode = stax.serial(
    stax.parallel(
        stax.serial(Dense(512), Relu),  # image
        # mu, logvar, gradmu, gradlogvar
        stax.serial(stax.FanInConcat(), Dense(256), Relu)
    ),
    stax.FanInConcat(),
    Dense(512), Relu,
    FanOut(2),
    stax.parallel(Dense(latent_dim),
                  Dense(latent_dim)),
)

decoder_init, decode = stax.serial(
    Dense(512), Relu,
    Dense(512), Relu,
    Dense(28 * 28),
)


def gaussian_kl(z_dist):
    """KL divergence from a diagonal Gaussian to the standard Gaussian."""
    mu, sigmasq = z_dist.mu, jax.nn.softplus(z_dist.logvar)
    return -0.5 * jnp.sum(1. + jnp.log(sigmasq) - mu**2. - sigmasq)


def gaussian_sample(rng, z_dist):
    """Sample a diagonal Gaussian."""
    mu, sigmasq = z_dist.mu, jax.nn.softplus(z_dist.logvar)
    return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)


def bernoulli_logpdf(logits, x):
    """Bernoulli log pdf of data x given logits."""
    return -jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits), )


def get_logits(rng, dec_params):
    return lambda z_dist: decode(dec_params, gaussian_sample(rng, z_dist))


def inner_elbo_fcn(rng, dec_params):
    def get_elbo(z_dist, images):
        logits_x = get_logits(rng, dec_params)(z_dist)
        return bernoulli_logpdf(logits_x, images) - gaussian_kl(z_dist)
    return get_elbo


def outer_loss(rng, params, images, num_steps, return_images=False):
    enc_params, dec_params, prior_mu_z, prior_logvar_z = params

    logvar_z_norms = []
    mu_z_norms = []
    elbos = []
    if(return_images):
        logit_images = [images]

    prior_mu_z = jnp.repeat(
        jnp.expand_dims(
            prior_mu_z,
            0),
        repeats=images.shape[0],
        axis=0)  # (B, D)
    prior_logvar_z = jnp.repeat(
        jnp.expand_dims(
            prior_logvar_z,
            0),
        repeats=images.shape[0],
        axis=0)  # (B, D)
    z_dist = GaussianParams(mu=prior_mu_z, logvar=prior_logvar_z)
    loss = 0
    for t in range(num_steps):

        elbo_fcn = inner_elbo_fcn(rng, dec_params)
        vmapped_value_and_grad = jax.vmap(jax.value_and_grad(elbo_fcn, 0))

        elbo, elbo_grad = vmapped_value_and_grad(z_dist, images)

        posterior_mu_z, posterior_logvar_z = encode(
            enc_params, (images, (z_dist.mu, z_dist.logvar, elbo_grad.mu, elbo_grad.logvar)))

        logvar_z_norm = jnp.linalg.norm(posterior_logvar_z[0])
        mu_z_norm = jnp.linalg.norm(posterior_mu_z[0])

        z_dist = GaussianParams(mu=posterior_mu_z, logvar=posterior_logvar_z)

        if(return_images):
            logits_x = get_logits(rng, dec_params)(z_dist)
            logit_images.append(jax.nn.sigmoid(logits_x))
        logvar_z_norms.append(logvar_z_norm)
        mu_z_norms.append(mu_z_norm)
        elbos.append(elbo[0])  # .primal)

        loss += jnp.sum(elbo)

    to_return = (loss / num_steps, logit_images, logvar_z_norms,
                 elbos) if return_images else loss / num_steps
    return to_return


def main():
    config.update("jax_debug_nans", True)
    step_size = 1e-4
    num_epochs = 200
    batch_size = 64
    nrow, ncol = 10, 10  # sampled image grid size
    num_steps = 4
    debug = False

    test_rng = random.PRNGKey(1)  # fixed prng key for evaluation
    vae_sample_dir = 'vae_samples'
    if not os.path.isdir(vae_sample_dir):
        os.makedirs(vae_sample_dir)
    imfile = os.path.join(vae_sample_dir, "mnist_vae_{:03d}.png")

    train_images, _, test_images, _ = datasets.mnist(permute_train=True)
    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)

    enc_init_rng, dec_init_rng, mu_init_rng = random.split(
        random.PRNGKey(0), 3)

    _, init_encoder_params = encoder_init(enc_init_rng, ((
        batch_size, 28 * 28), [(batch_size, latent_dim) for i in range(4)]))

    _, init_decoder_params = decoder_init(
        dec_init_rng, (batch_size, latent_dim))
    prior_mu_z = random.normal(mu_init_rng, (latent_dim,))

    prior_logvar_z = jnp.zeros((latent_dim,))
    init_params = init_encoder_params, init_decoder_params, prior_mu_z, prior_logvar_z

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)

    train_images = jax.device_put(train_images)
    test_images = jax.device_put(test_images)

    def binarize_batch(rng, i, images):
        i = i % num_batches
        batch = lax.dynamic_slice_in_dim(
            images, i * batch_size, batch_size)
        return random.bernoulli(rng, batch)

    @jit
    def run_epoch(rng, opt_state, images):
        def body_fun(i, opt_state):
            elbo_rng, data_rng = random.split(random.fold_in(rng, i))
            batch = binarize_batch(data_rng, i, images)

            def loss(params): return -outer_loss(elbo_rng,
                                                 params, batch, num_steps) / batch_size
            g = grad(loss)(get_params(opt_state))
            return opt_update(i, g, opt_state)
        return lax.fori_loop(0, num_batches, body_fun, opt_state)

    def evaluate(opt_state, images):
        params = get_params(opt_state)
        elbo_rng, data_rng, image_rng = random.split(test_rng, 3)
        num_test = 10
        binarized_test = random.bernoulli(data_rng, images[:num_test, :])
        test_elbo, logit_images, logvar_z_norms, elbos = outer_loss(
            elbo_rng, params, binarized_test, num_steps, True)
        stack_logit_images = jnp.stack(logit_images)
        sampled_logit_images = image_grid(
            num_steps + 1, num_test, stack_logit_images, (28, 28))
        return -test_elbo / \
            images.shape[0], sampled_logit_images, logvar_z_norms, elbos

    def image_grid(nrow, ncol, imagevecs, imshape):
        """Reshape a stack of image vectors into an image grid for plotting."""
        images = iter(imagevecs.reshape((-1,) + imshape))
        return [[next(images).T for _ in range(ncol)][::-1]
                for _ in range(nrow)]

    opt_state = opt_init(init_params)
    for epoch in range(num_epochs):
        tic = time.time()
        if not debug:
            opt_state = run_epoch(
                random.PRNGKey(epoch), opt_state, train_images)
            test_elbo, logit_images, logvar_norms, elbos = jit(
                evaluate)(opt_state, test_images)
        else:
            rng = random.PRNGKey(epoch)
            for i in range(num_batches):
                elbo_rng, data_rng = random.split(random.fold_in(rng, i))
                batch = binarize_batch(data_rng, i, train_images)

                def loss(params): return -outer_loss(elbo_rng,
                                                     params, batch, num_steps) / batch_size

                g = grad(loss)(get_params(opt_state))
                opt_state = opt_update(i, g, opt_state)

            test_elbo, logit_images, logvar_norms, elbos = evaluate(
                opt_state, test_images)

        fig, axs = plt.subplots(
            len(logit_images[0]),
            len(logit_images),
            figsize=(2 * len(logit_images), 2 * len(logit_images[0])),
            constrained_layout=True)
        fig.suptitle('Training Epoch: ' + str(epoch), fontsize=36)
        for i in range(len(logit_images[0])):
            for j in range(len(logit_images)):
                axs[i][j].imshow(logit_images[j][i].T, cmap=plt.cm.gray)
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])

        axs[0][0].set_title('Ground Truth', fontsize=20)
        for k in range(1, len(logit_images)):
            axs[0][k].set_title('Iter {}'.format(k), fontsize=20)

        plt.savefig(imfile.format(epoch))
        plt.close()


if __name__ == "__main__":
    main()
