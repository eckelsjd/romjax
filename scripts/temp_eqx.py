import gzip
import struct
from pathlib import Path
from urllib.request import urlretrieve

import equinox as eqx
import jax.numpy as jnp
import jax
import optax
import numpy as np

import matplotlib.pyplot as plt

from romjax.plotting import gridplot
from romjax.optimization import Optimizer

jax.config.update('jax_platform_name', 'cpu')



def simple_mlp():
    def fun(x):
        return jnp.sin(jnp.pi * x) + x ** 2

    log_interval = 100
    num_steps = 2000

    ntrain = 15
    ntest = 5
    key = jax.random.PRNGKey(0)
    test_key = jax.random.PRNGKey(3)
    nn_key = jax.random.PRNGKey(1)

    # Use explicit (n, 1) shapes so the MLP input/output shapes are unambiguous.
    x = jax.random.uniform(key, (ntrain, 1))
    y = fun(x)
    xlin = jnp.linspace(0, 1, 200).reshape(-1, 1)
    ylin = fun(xlin)
    xtest = jax.random.uniform(test_key, (ntest, 1))
    ytest = fun(xtest)

    width = 12
    depth = 2
    lr = 0.01
    model = eqx.nn.MLP(1, 1, width, depth, key=nn_key, activation=jax.nn.sigmoid)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    print(model)

    @eqx.filter_jit
    def loss_fn(model, x, y):
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y)**2)

    @eqx.filter_jit
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for i in range(num_steps):
        model, opt_state, loss = step(model, opt_state, x, y)

        if i % log_interval == 0:
            test = loss_fn(model, xtest, ytest)
            print(f"i={i}, loss={loss}, test={test}")

    pred_y = jax.vmap(model)(x)
    pred_ylin = jax.vmap(model)(xlin)
    pred_ytest = jax.vmap(model)(xtest)

    fig, ax = plt.subplots()
    ax.plot(x[:, 0], y[:, 0], '.k', ms=10, label='Train')
    ax.plot(xlin[:, 0], ylin[:, 0], '-k', lw=2, label='True')
    ax.plot(x[:, 0], pred_y[:, 0], '.r', ms=8)
    ax.plot(xlin[:, 0], pred_ylin[:, 0], '--r', lw=1, label='NN')
    ax.plot(xtest[:, 0], pred_ytest[:, 0], '.b', ms=10, label='Test')
    ax.legend()
    plt.show()


def simple_cnn():
    base_urls = (
        "https://storage.googleapis.com/cvdf-datasets/mnist",
        "https://yann.lecun.com/exdb/mnist",
        "http://yann.lecun.com/exdb/mnist",
    )
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    cache_dir = Path.home() / ".cache" / "romjax" / "mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in files.values():
        path = cache_dir / filename
        if path.exists():
            continue
        last_err: Exception | None = None
        for base_url in base_urls:
            try:
                urlretrieve(f"{base_url}/{filename}", path)
                last_err = None
                break
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
        if last_err is not None:
            raise RuntimeError(f"Failed to download {filename}") from last_err

    def load_images(path: Path) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            data = f.read()
        magic, num, rows, cols = struct.unpack(">IIII", data[:16])
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(num, rows, cols)
        return jnp.asarray(images.astype(np.float32) / 255.0)

    def load_labels(path: Path) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            data = f.read()
        magic, num = struct.unpack(">II", data[:8])
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        return np.frombuffer(data, dtype=np.uint8, offset=8)

    x_train = load_images(cache_dir / files["train_images"])[:, jnp.newaxis, ...]  # (B, C, W, H)
    y_train = load_labels(cache_dir / files["train_labels"])
    x_test = load_images(cache_dir / files["test_images"])[:, jnp.newaxis, ...]
    y_test = load_labels(cache_dir / files["test_labels"])

    print(
        "MNIST loaded: "
        f"x_train={x_train.shape}, y_train={y_train.shape}, "
        f"x_test={x_test.shape}, y_test={y_test.shape}"
    )

    BATCH_SIZE = 1024
    LEARNING_RATE = 0.015
    STEPS = 2000
    PRINT_EVERY = 100
    PLOT_EVERY = 25
    SEED = 5678
    EPOCHS = 30

    key = jax.random.PRNGKey(SEED)

    def dataloader(arrays, batch_size, epochs=-1):
        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = np.arange(dataset_size)
        cnt_epochs = 0
        while True:
            perm = np.random.permutation(indices)
            start = 0
            end = batch_size
            while end <= dataset_size:
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end
                end = start + batch_size
            cnt_epochs += 1

            if epochs > 0 and cnt_epochs >= epochs:
                break

    class CNN(eqx.Module):
        layers: list

        def __init__(self, key):
            key1, key2, key3, key4 = jax.random.split(key, 4)
            self.layers = [
                eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                eqx.nn.MaxPool2d(kernel_size=2),
                jax.nn.relu,
                jnp.ravel,
                eqx.nn.Linear(1728, 512, key=key2),
                jax.nn.sigmoid,
                eqx.nn.Linear(512, 64, key=key3),
                jax.nn.relu,
                eqx.nn.Linear(64, 10, key=key4),
                jax.nn.log_softmax,
            ]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    def cross_entropy(y, pred_y):
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        return -jnp.mean(pred_y)

    @eqx.filter_jit
    def loss(model, args):
        x, y = args
        pred_y = jax.vmap(model)(x)
        return cross_entropy(y, pred_y)
    
    @eqx.filter_jit
    def compute_accuracy(model, x, y):
        pred_y = jax.vmap(model)(x)
        pred_y = jnp.argmax(pred_y, axis=1)
        return jnp.mean(y == pred_y)
    
    @eqx.filter_jit
    def evaluate(model):
        testloader = dataloader((x_test, y_test), BATCH_SIZE, epochs=1)
        avg_acc = 0
        cnt = 0
        for x, y in testloader:
            avg_acc += compute_accuracy(model, x, y)
            cnt += 1
        return avg_acc / cnt

    opt = Optimizer()
    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)
    trainloader = dataloader((x_train, y_train), BATCH_SIZE, epochs=EPOCHS)  # infinite

    # def train(model, trainloader, optim, steps, print_every):
    #     opt_state = optim.init(eqx.filter(model, eqx.is_array))

    #     @eqx.filter_jit
    #     def make_step(model, opt_state, args):
    #         loss_value, grads = eqx.filter_value_and_grad(loss)(model, args)
    #         updates, opt_state = optim.update(
    #             grads, opt_state, eqx.filter(model, eqx.is_array)
    #         )
    #         model = eqx.apply_updates(model, updates)
    #         return model, opt_state, loss_value

    #     for step, (x, y) in zip(range(steps), trainloader):
    #         model, opt_state, train_loss = make_step(model, opt_state, (x,y))
    #         if (step % print_every) == 0 or (step == steps - 1):
    #             test_accuracy = evaluate(model)
    #             print(
    #                 f"{step=}, train_loss={train_loss.item()}, test_accuracy={test_accuracy.item()}"
    #             )
    #     return model
    
    # model_hat = train(model, trainloader, optax.adamw(LEARNING_RATE), STEPS, PRINT_EVERY)

    model_hat = opt.run_debug(
        loss,
        model,
        optax.adam(LEARNING_RATE),
        trainloader,
        max_steps=STEPS,
        grad_tol=-1e-20,
        param_tol=-1e-20,
        loss_tol=-1e-20,
        log_interval=PRINT_EVERY,
        plot_interval=PLOT_EVERY,
        # test_fn=evaluate,
        live_plot=True
    )

    print(f"Final test accuracy: {evaluate(model_hat)}")

    W, H = x_train.shape[2:]
    xg, yg = jnp.meshgrid(jnp.linspace(0, 1, W), jnp.linspace(0, 1, H))
    nshow = 9
    indices = np.random.randint(0, len(y_test)-1, nshow)
    plot_specs = [
        {
            'kind': 'pcolor',
            'data': (xg, yg, jnp.flip(x_test[i, 0, :, :], axis=0)),
            'opts': {'ax_visible': False},
            'kwargs': {'cmap': 'gray'}
        }
        for i in indices
    ]
    def _adjust(fig, axs, *args):
        axs = axs.flatten()
        for ax_idx, test_idx in enumerate(indices):
            true_label = int(y_test[test_idx])
            pred_y = model_hat(x_test[test_idx, :, :])
            pred_label = int(jnp.argmax(pred_y))
            axs[ax_idx].set_title(f"True: {true_label:d}, Pred: {pred_label:d}", color='white')

    fig, ax = gridplot(plot_specs, scheme='dark', adjust=_adjust)
    plt.show()


if __name__ == '__main__':
    # simple_mlp()
    simple_cnn()
