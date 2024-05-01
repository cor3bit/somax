"""
NanoLM example from Optax
https://optax.readthedocs.io/en/latest/_collections/examples/nanolm.html
"""

import jax
import jax.numpy as jnp
import optax

from matplotlib import pyplot as plt
import tensorflow_datasets as tfds

from benchmarks.utils.model_zoo import NanoLM

# platform check
print("JAX running on", jax.devices()[0].platform.upper())

# @markdown Random seed:
SEED = 42  # @param{type:"integer"}
# @markdown Learning rate passed to the optimizer:
LEARNING_RATE = 5e-3  # @param{type:"number"}
# @markdown Batch size:
BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown Numer of training iterations:
N_ITERATIONS = 50_000  # @param{type:"integer"}
# @markdown Number of training iterations between two consecutive evaluations:
N_FREQ_EVAL = 2_000  # @param{type:"integer"}
# @markdown Rate for dropout in the transformer model
DROPOUT_RATE = 0.2  # @param{type:"number"}
# @markdown Context window for the transformer model
BLOCK_SIZE = 64  # @param{type:"integer"}
# @markdown Number of layer for the transformer model
NUM_LAYERS = 6  # @param{type:"integer"}
# @markdown Size of the embedding for the transformer model
EMBED_SIZE = 256  # @param{type:"integer"}
# @markdown Number of heads for the transformer model
NUM_HEADS = 8  # @param{type:"integer"}
# @markdown Size of the heads for the transformer model
HEAD_SIZE = 32  # @param{type:"integer"}

ds = tfds.load("tiny_shakespeare")

# combine train and test examples into a single string
text_train = ""
for example in ds["train"].concatenate(ds["test"]).as_numpy_iterator():
    text_train += example["text"].decode("utf-8")

# similarly, create a single string for validation
text_validation = ""
for example in ds["validation"].as_numpy_iterator():
    text_validation += example["text"].decode("utf-8")

print(f"Length of text for training: {len(text_train):_} characters")
print(f"Length of text for validation: {len(text_validation):_} characters")

# small sample of the train set
print(text_train[:1000])

vocab = sorted(list(set(text_train)))
print("Vocabulary:, ", "".join(vocab))
print("Length of vocabulary: ", len(vocab))

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# encode train and validation data
train_data = jnp.array(encode(text_train))
eval_data = jnp.array(encode(text_validation))

dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))


@jax.jit
def get_batch(random_key, data):
    """Prepares a random batch of training data.

    Args:
        random_key: A random seed for sampling a batch.
        data: The complete training dataset.

    Returns:
        x: Input sequences.
        y: Target sequences (shifted inputs).
    """
    ix = jax.random.randint(
        random_key, shape=(BATCH_SIZE, 1), minval=0, maxval=len(data) - BLOCK_SIZE
    )
    x = dynamic_slice_vmap(data, ix, (BLOCK_SIZE,))
    y = dynamic_slice_vmap(data, ix + 1, (BLOCK_SIZE,))
    return x, y


model = NanoLM(
    vocab_size=len(vocab),
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    head_size=HEAD_SIZE,
    dropout_rate=DROPOUT_RATE,
    embed_size=EMBED_SIZE,
    block_size=BLOCK_SIZE,
)


def loss_fun(params, x, y, dropout_key):
    logits = model.apply(params, x, training=True, rngs={"dropout": dropout_key})
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()


@jax.jit
def eval_step(params, x, y):
    logits = model.apply(params, x, training=False)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=y
    ).mean()


key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)

var_params = model.init(
    key,
    jnp.ones((BATCH_SIZE, BLOCK_SIZE), dtype=jnp.int32),
    training=False,
)

n_params = sum(p.size for p in jax.tree_util.tree_leaves(var_params))

print(f"Total number of parameters: {n_params:_}")

# To run with SGD instead of adam, replace `adam` with `sgd`
opt = optax.adamw(learning_rate=LEARNING_RATE)

opt_state = opt.init(var_params)

all_train_losses = []
all_eval_losses = []


# we define one iteration of the optimizer and JIT this function
@jax.jit
def step(key, params, opt_state):
    key, subkey = jax.random.split(key)
    batch = get_batch(key, train_data)
    loss, grad = jax.value_and_grad(loss_fun)(params, *batch, subkey)
    updates, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, key, opt_state, loss


for i in range(N_ITERATIONS):
    var_params, key, opt_state, loss = step(key, var_params, opt_state)
    all_train_losses.append(loss)

    # once every N_FREQ_EVAL we compute loss on the validation set
    if i % N_FREQ_EVAL == 0:
        key, subkey = jax.random.split(key)
        eval_loss = eval_step(var_params, *get_batch(subkey, eval_data))
        all_eval_losses.append(eval_loss)
        print(f"Step: {i}\t train loss: {loss}\t eval loss: {eval_loss}")

plt.title(f"Convergence of adamw (train loss)")
plt.plot(all_train_losses, label="train", lw=3)
plt.plot(
    jnp.arange(0, len(all_eval_losses) * N_FREQ_EVAL, N_FREQ_EVAL),
    all_eval_losses,
    label="test",
    lw=3,
)
plt.xlabel("steps")
plt.ylabel("loss")
plt.grid()
plt.legend(frameon=False)
plt.show()

# Let's now generate some text
key, subkey = jax.random.split(key)
text = model.generate(key, var_params, 1000)[:, 0, 0].tolist()
print(decode(text))
