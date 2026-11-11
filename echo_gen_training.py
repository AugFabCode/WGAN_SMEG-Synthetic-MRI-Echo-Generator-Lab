"""Original CEASET architecture, sampling, WGAN-GP loss and training loop."""
from echo_gen_preprocessing import *
from echo_gen_preview import PreviewWriter, axial_chunk_axis
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


from echo_gen_control import check_training_stop

def configure_training_devices():
    requested = PIPELINE_ACTION in {'train', 'all'}
    if TRAINING_DEVICE_MODE not in {'dual_gpu', 'single_gpu', 'cpu'}:
        raise ValueError(f'Unknown TRAINING_DEVICE_MODE: {TRAINING_DEVICE_MODE}')
    mode = TRAINING_DEVICE_MODE if requested else 'cpu'
    # get_visible_devices also respects GPUs hidden by an earlier tf.config call.
    # This query does not initialize logical devices or allocate GPU memory.
    visible = tf.config.get_visible_devices('GPU')
    needed = {'dual_gpu': 2, 'single_gpu': 1, 'cpu': 0}[mode]
    if len(visible) < needed:
        raise RuntimeError(
            f'{mode} requires {needed} TensorFlow-visible GPU(s), but found '
            f'{len(visible)}. Python: {sys.executable}. '
            'Check the selected kernel and CUDA libraries; GPU training has not started.'
        )

    if needed:
        for gpu in visible:
            # Reuse an existing compatible configuration, including after a
            # previous training cell. Do not silently ignore incompatible state.
            try:
                if tf.config.experimental.get_memory_growth(gpu) is not True:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f'{gpu.name}: TensorFlow has already initialized the runtime '
                    'without the required memory-growth setting. Restart the kernel '
                    'and run the notebook in order before creating tensors or models. '
                    'Importing TensorFlow or listing physical GPUs alone is allowed.'
                ) from exc
            except ValueError as exc:
                raise RuntimeError(
                    f'{gpu.name}: the existing GPU configuration is incompatible '
                    'with memory growth (for example, logical-device memory limits). '
                    "Use a fresh kernel with this notebook's device setup."
                ) from exc

    devices = {
        'dual_gpu': ('/GPU:0', '/GPU:1'),
        'single_gpu': ('/GPU:0', '/GPU:0'),
        'cpu': ('/CPU:0', '/CPU:0'),
    }[mode]
    keras.mixed_precision.set_global_policy('float32')
    tf.keras.utils.set_random_seed(TRAINING_SEED)
    print('TensorFlow:', tf.__version__, '| visible GPUs:', len(visible),
          '| generator / critic:', devices, flush=True)
    return devices


GENERATOR_DEVICE, CRITIC_DEVICE = configure_training_devices()


def assert_training_hardware():
    expected = {'dual_gpu': ('/GPU:0','/GPU:1'),
                'single_gpu': ('/GPU:0','/GPU:0'), 'cpu': ('/CPU:0','/CPU:0')}
    if (GENERATOR_DEVICE, CRITIC_DEVICE) != expected[TRAINING_DEVICE_MODE]:
        raise RuntimeError('Change PIPELINE_ACTION and restart/run from the top before training.')
    if TRAINING_ENABLE_PERCEPTUAL_LOSS and not RADIMAGENET_RESNET50_WEIGHTS.is_file():
        raise FileNotFoundError(f'Perceptual loss enabled but weights are missing: {RADIMAGENET_RESNET50_WEIGHTS}')
    if TRAINING_DEVICE_MODE == 'single_gpu':
        print('[MEMORY] Generator and WGAN-GP critic share one GPU; reduce chunk size if needed.')
    if TRAINING_DEVICE_MODE == 'cpu':
        print('[CPU] Suitable for small smoke tests; full 3-D training will be slow.')


class InstanceNorm3D(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1,2,3], keepdims=True)
        x = (x - mean) * tf.math.rsqrt(var + self.epsilon)
        return x * self.gamma[None,None,None,None,:] + self.beta[None,None,None,None,:]


class TEEmbedding(layers.Layer):
    def __init__(self, dim=MODEL_TE_EMBED_DIM, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(dim, activation=tf.nn.silu)
        self.d2 = layers.Dense(dim, activation=tf.nn.silu)

    def call(self, te_ms):
        te = tf.cast(te_ms, tf.float32) / float(TE_EMBEDDING_SCALE_MS)
        if te.shape.rank == 1:
            te = te[:, None]
        return self.d2(self.d1(te))


class FiLM3D(layers.Layer):
    """
    Learned FiLM + explicit low-order raw-TE side path.

    The original learned embedding path is unchanged.  V7 adds a separate
    raw-TE polynomial [t, t^2] -> (gamma,beta) path.  It is deliberately
    low-order/smooth and zero-initialized, so migrating a legacy checkpoint
    initially reproduces the old network exactly.
    """
    def __init__(self, channels: int, use_raw_te: bool = RAW_TE_FILM_ENABLED, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.use_raw_te = bool(use_raw_te)
        self.affine = layers.Dense(
            2 * self.channels,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="learned_embedding_affine",
        )
        self.raw_te_affine = None
        if self.use_raw_te:
            self.raw_te_affine = layers.Dense(
                2 * self.channels,
                use_bias=False,
                kernel_initializer="zeros",
                name="raw_te_polynomial_affine",
            )

    def _raw_te_features(self, te_ms):
        t = tf.cast(te_ms, tf.float32) / float(RAW_TE_PHYSICAL_SCALE_MS)
        if t.shape.rank == 1:
            t = t[:, None]
        feats = [t]
        if int(RAW_TE_FILM_POLYNOMIAL_ORDER) >= 2:
            feats.append(tf.square(t))
        return tf.concat(feats, axis=-1)

    def call(self, x, conditioning, raw_te_ms=None):
        params = self.affine(conditioning)
        if self.use_raw_te and raw_te_ms is not None:
            params = params + self.raw_te_affine(
                self._raw_te_features(raw_te_ms)
            )
        gamma, beta = tf.split(params, 2, axis=-1)
        gamma = gamma[:,None,None,None,:]
        beta = beta[:,None,None,None,:]
        return x * (1.0 + gamma) + beta


class ConditionalResidualBlock3D(layers.Layer):
    def __init__(self, filters, stride=1, l2=MODEL_L2, name=None, use_raw_te=RAW_TE_FILM_ENABLED):
        super().__init__(name=name)
        self.filters = int(filters)
        self.stride = int(stride)
        self.use_raw_te = bool(use_raw_te)
        reg = regularizers.l2(l2) if l2 else None
        self.conv1 = layers.Conv3D(filters, 3, strides=stride, padding="same", kernel_initializer="he_normal", kernel_regularizer=reg)
        self.norm1 = InstanceNorm3D()
        self.film1 = FiLM3D(filters, use_raw_te=self.use_raw_te, name="film1")
        self.act1 = layers.LeakyReLU(MODEL_LEAKY_RELU)
        self.conv2 = layers.Conv3D(filters, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=reg)
        self.norm2 = InstanceNorm3D()
        self.film2 = FiLM3D(filters, use_raw_te=self.use_raw_te, name="film2")
        self.act2 = layers.LeakyReLU(MODEL_LEAKY_RELU)
        self.proj = None

    def build(self, input_shape):
        channels = int(input_shape[0][-1] if isinstance(input_shape, (tuple,list)) and isinstance(input_shape[0], (tuple,list,tf.TensorShape)) else input_shape[-1])
        if channels != self.filters or self.stride != 1:
            self.proj = layers.Conv3D(self.filters, 1, strides=self.stride, padding="same", kernel_initializer="he_normal")
        super().build(input_shape)

    def call(self, inputs, training=False):
        if len(inputs) == 3:
            x, conditioning, raw_te_ms = inputs
        else:
            x, conditioning = inputs
            raw_te_ms = None
        shortcut = x if self.proj is None else self.proj(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.film1(y, conditioning, raw_te_ms=raw_te_ms)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.film2(y, conditioning, raw_te_ms=raw_te_ms)
        return self.act2(shortcut + y)


class TargetAwareSetFusion(layers.Layer):
    """Permutation-invariant masked attention over the echo axis."""
    def __init__(self, hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.d1 = layers.Dense(hidden, activation=tf.nn.silu)
        self.d2 = layers.Dense(1, activation=None)

    def call(self, features, input_te_emb, target_te_emb, echo_times_ms, target_te_ms, echo_mask):
        # features: [B,K,H,W,D,C]
        pooled = tf.reduce_mean(features, axis=[2,3,4])
        k = tf.shape(features)[1]
        target_rep = tf.repeat(target_te_emb[:,None,:], k, axis=1)
        # Give the set-attention gate the physical relationship explicitly in addition to learned embeddings.
        # Padding TEs are harmless because their logits are masked before softmax.
        input_te_scalar = tf.cast(echo_times_ms[...,None], tf.float32) / float(TE_EMBEDDING_SCALE_MS)
        target_te_scalar = tf.repeat(
            tf.cast(target_te_ms[:,None,:], tf.float32) / float(TE_EMBEDDING_SCALE_MS),
            k, axis=1,
        )
        delta_te = target_te_scalar - input_te_scalar
        x = tf.concat([
            pooled,
            input_te_emb,
            target_rep,
            input_te_scalar,
            target_te_scalar,
            delta_te,
            tf.abs(delta_te),
        ], axis=-1)
        logits = tf.squeeze(self.d2(self.d1(x)), axis=-1)
        mask = tf.cast(echo_mask > 0.5, logits.dtype)
        logits = tf.where(mask > 0, logits, tf.fill(tf.shape(logits), tf.cast(-1e9, logits.dtype)))
        weights = tf.nn.softmax(logits, axis=1)
        weights = weights * mask
        weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-6)
        expand = weights[:, :, None, None, None, None]
        fused = tf.reduce_sum(features * expand, axis=1)
        return fused, weights


class EchoSetFiLMGenerator(keras.Model):
    def __init__(self, modality: str, filters=MODEL_FILTERS, use_raw_te_film=RAW_TE_FILM_ENABLED, **kwargs):
        super().__init__(name=f"ceaset_generator_{modality}", **kwargs)
        if modality not in {"magnitude", "phase"}:
            raise ValueError(modality)
        self.modality = modality
        self.filters = tuple(map(int, filters))
        self.use_raw_te_film = bool(use_raw_te_film)
        self.te_embedding = TEEmbedding(MODEL_TE_EMBED_DIM, name="te_embedding")
        self.encoder_blocks = [
            ConditionalResidualBlock3D(self.filters[0], stride=1, name="enc1", use_raw_te=self.use_raw_te_film),
            ConditionalResidualBlock3D(self.filters[1], stride=2, name="enc2", use_raw_te=self.use_raw_te_film),
            ConditionalResidualBlock3D(self.filters[2], stride=2, name="enc3", use_raw_te=self.use_raw_te_film),
            ConditionalResidualBlock3D(self.filters[3], stride=2, name="enc4", use_raw_te=self.use_raw_te_film),
        ]
        self.fusions = [TargetAwareSetFusion(name=f"fusion{i+1}") for i in range(4)]
        self.target_bottleneck = ConditionalResidualBlock3D(self.filters[3], stride=1, name="target_bottleneck", use_raw_te=self.use_raw_te_film)
        self.up3 = layers.UpSampling3D(2, name="up3")
        self.up3_conv = layers.Conv3D(self.filters[2], 3, padding="same", kernel_initializer="he_normal")
        self.dec3 = ConditionalResidualBlock3D(self.filters[2], name="dec3", use_raw_te=self.use_raw_te_film)
        self.up2 = layers.UpSampling3D(2, name="up2")
        self.up2_conv = layers.Conv3D(self.filters[1], 3, padding="same", kernel_initializer="he_normal")
        self.dec2 = ConditionalResidualBlock3D(self.filters[1], name="dec2", use_raw_te=self.use_raw_te_film)
        self.up1 = layers.UpSampling3D(2, name="up1")
        self.up1_conv = layers.Conv3D(self.filters[0], 3, padding="same", kernel_initializer="he_normal")
        self.dec1 = ConditionalResidualBlock3D(self.filters[0], name="dec1", use_raw_te=self.use_raw_te_film)
        self.final_fuse = layers.Conv3D(self.filters[0], 3, padding="same", activation=tf.nn.silu, kernel_initializer="he_normal")
        # Zero initialization means prediction == deterministic physics baseline before training.
        # V7 no longer penalizes this residual voxel-by-voxel, so it is free to
        # correct vessels/lesions/rims even when the baseline is locally wrong.
        self.residual_head = layers.Conv3D(1, 3, padding="same", kernel_initializer="zeros", bias_initializer="zeros", name="residual_head")

    def _encode_set(self, echo_images, echo_times_ms, echo_mask, target_te_emb, target_te_ms, training=False):
        shape = tf.shape(echo_images)
        b, k = shape[0], shape[1]
        x = tf.reshape(echo_images, [b*k, shape[2], shape[3], shape[4], shape[5]])
        input_te_flat = tf.reshape(echo_times_ms, [b*k, 1])
        input_te_emb_flat = self.te_embedding(input_te_flat)
        input_te_emb = tf.reshape(input_te_emb_flat, [b, k, -1])
        fused_levels=[]
        attention=[]
        for block, fusion in zip(self.encoder_blocks, self.fusions):
            x = block([x, input_te_emb_flat, input_te_flat], training=training)
            xs = tf.shape(x)
            x_set = tf.reshape(x, [b, k, xs[1], xs[2], xs[3], xs[4]])
            fused, weights = fusion(x_set, input_te_emb, target_te_emb, echo_times_ms, target_te_ms, echo_mask)
            fused_levels.append(fused)
            attention.append(weights)
        return fused_levels, attention

    def call(self, inputs, training=False, return_aux=False):
        echo_images = tf.cast(inputs["echo_images"], tf.float32)
        echo_times_ms = tf.cast(inputs["echo_times_ms"], tf.float32)
        echo_mask = tf.cast(inputs["echo_mask"], tf.float32)
        target_te_ms = tf.cast(inputs["target_te_ms"], tf.float32)
        baseline = tf.cast(inputs["physics_baseline"], tf.float32)

        target_te_emb = self.te_embedding(target_te_ms)
        levels, attention = self._encode_set(echo_images, echo_times_ms, echo_mask, target_te_emb, target_te_ms, training=training)
        f1,f2,f3,f4 = levels
        x = self.target_bottleneck([f4, target_te_emb, target_te_ms], training=training)
        x = self.up3_conv(self.up3(x))
        x = tf.concat([x, f3], axis=-1)
        x = self.dec3([x, target_te_emb, target_te_ms], training=training)
        x = self.up2_conv(self.up2(x))
        x = tf.concat([x, f2], axis=-1)
        x = self.dec2([x, target_te_emb, target_te_ms], training=training)
        x = self.up1_conv(self.up1(x))
        x = tf.concat([x, f1, baseline], axis=-1)
        x = self.dec1([x, target_te_emb, target_te_ms], training=training)
        x = self.final_fuse(x)
        residual = self.residual_head(x)
        prediction = baseline + residual
        if self.modality == "magnitude":
            prediction = tf.nn.relu(prediction)
        if "brain_mask" in inputs:
            prediction = prediction * tf.cast(inputs["brain_mask"], prediction.dtype)
            residual = residual * tf.cast(inputs["brain_mask"], residual.dtype)
        if return_aux:
            return prediction, {"residual": residual, "attention": attention}
        return prediction


class TEConditionedCritic(keras.Model):
    """
    Lightweight TE-conditioned critic for dual-GPU WGAN-GP training.

    Important design choices:
      - no InstanceNorm / BatchNorm in the critic;
      - no per-resolution FiLM blocks;
      - four strided 3-D convolutions only;
      - target TE is injected after global pooling;
      - the deterministic physics baseline is concatenated with the candidate
        image at the input so the critic still judges residual realism.

    The generator is intentionally unchanged. The critic is reset when
    transitioning from a weight-clipped or incompatible historical checkpoint.
    """
    def __init__(self, **kwargs):
        super().__init__(name="ceaset_te_conditioned_critic_gp", **kwargs)
        self.te_embedding = TEEmbedding(32, name="critic_te_embedding")
        self.convs = [
            layers.Conv3D(8, 3, strides=2, padding="same",
                          kernel_initializer="he_normal", name="d_conv1"),
            layers.Conv3D(16, 3, strides=2, padding="same",
                          kernel_initializer="he_normal", name="d_conv2"),
            layers.Conv3D(32, 3, strides=2, padding="same",
                          kernel_initializer="he_normal", name="d_conv3"),
            layers.Conv3D(48, 3, strides=2, padding="same",
                          kernel_initializer="he_normal", name="d_conv4"),
        ]
        self.acts = [
            layers.LeakyReLU(MODEL_LEAKY_RELU, name=f"d_lrelu{i+1}")
            for i in range(4)
        ]
        self.gap = layers.GlobalAveragePooling3D(name="d_gap")
        self.dense = layers.Dense(64, activation=tf.nn.silu, name="d_dense")
        self.out = layers.Dense(1, name="d_score")

    def call(self, inputs, training=False):
        image = tf.cast(inputs["image"], tf.float32)
        baseline = tf.cast(
            inputs.get("physics_baseline", tf.zeros_like(image)),
            tf.float32,
        )
        target_te = tf.cast(inputs["target_te_ms"], tf.float32)

        x = tf.concat([image, baseline], axis=-1)
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x))

        x = self.gap(x)
        te = self.te_embedding(target_te)
        x = tf.concat([x, te], axis=-1)
        return self.out(self.dense(x))


def _generator_dummy_inputs(chunk_shape):
    h, w, d, _ = chunk_shape
    return {
        "echo_images": tf.zeros([1, 1, h, w, d, 1], dtype=tf.float32),
        "echo_times_ms": tf.constant([[4.3]], dtype=tf.float32),
        "echo_mask": tf.constant([[1.0]], dtype=tf.float32),
        "target_te_ms": tf.constant([[21.5]], dtype=tf.float32),
        "physics_baseline": tf.zeros([1, h, w, d, 1], dtype=tf.float32),
        "brain_mask": tf.ones([1, h, w, d, 1], dtype=tf.float32),
    }


def build_ceaset_generator_only(
    modality: str,
    chunk_shape: tuple[int, int, int, int],
    *,
    use_raw_te_film: bool = RAW_TE_FILM_ENABLED,
):
    with tf.device(GENERATOR_DEVICE):
        generator = EchoSetFiLMGenerator(
            modality,
            use_raw_te_film=bool(use_raw_te_film),
        )
        _ = generator(_generator_dummy_inputs(chunk_shape), training=False)
    return generator


def _weight_key(variable) -> str:
    name = getattr(variable, "path", None) or getattr(variable, "name", "")
    name = str(name).replace(":0", "")
    parts = name.split("/")
    if parts and parts[0].startswith("ceaset_generator_"):
        parts = parts[1:]
    return "/".join(parts)


def load_generator_weights_v7_compatible(
    target_generator,
    checkpoint_path: str | Path,
    modality: str,
    chunk_shape: tuple[int, int, int, int],
) -> str:
    """
    Load a native V7 generator if possible.  If the checkpoint is a legacy
    pre-raw-TE generator, instantiate the old topology, load it strictly, then
    copy every matching variable into V7.  Only the new raw-TE affine weights
    remain at their zero initialization.
    """
    checkpoint_path = Path(checkpoint_path)

    try:
        target_generator.load_weights(checkpoint_path)
        return "native_v7"
    except Exception as native_exc:
        print(
            f"[MIGRATE] {modality}: checkpoint is not native V7; "
            "attempting controlled legacy-generator migration.",
            flush=True,
        )

    legacy = build_ceaset_generator_only(
        modality,
        chunk_shape,
        use_raw_te_film=False,
    )
    legacy.load_weights(checkpoint_path)

    # Preserve topology rather than relying on Keras-generated variable names:
    # filtering the explicitly new raw-TE kernels from V7 must leave exactly the
    # legacy variable sequence and shapes.
    target_legacy_vars = [
        v
        for v in target_generator.weights
        if "raw_te_polynomial_affine" not in _weight_key(v)
    ]
    if len(target_legacy_vars) != len(legacy.weights):
        raise RuntimeError(
            "Legacy migration topology mismatch: "
            f"legacy={len(legacy.weights)} variables, "
            f"V7-compatible={len(target_legacy_vars)} variables."
        )

    copied = 0
    for index, (src_var, dst_var) in enumerate(
        zip(legacy.weights, target_legacy_vars)
    ):
        if tuple(src_var.shape) != tuple(dst_var.shape):
            raise RuntimeError(
                "Legacy migration shape mismatch at variable "
                f"{index}: legacy {_weight_key(src_var)} {tuple(src_var.shape)} "
                f"vs V7 {_weight_key(dst_var)} {tuple(dst_var.shape)}"
            )
        dst_var.assign(src_var)
        copied += 1

    print(
        f"[MIGRATE] {modality}: copied {copied} legacy variables into V7; "
        "new raw-TE FiLM paths remain zero-initialized.",
        flush=True,
    )
    del legacy
    gc.collect()
    return "legacy_migrated_to_v7"


def build_ceaset_models(modality: str, chunk_shape: tuple[int,int,int,int]):
    """
    Build the generator on GPU:0 and critic on GPU:1.

    A K=1 dummy is sufficient to instantiate every shared generator weight.
    Using K=1 here avoids the unnecessary K=4 construction-time activation
    spike that the earlier notebook incurred.
    """
    h, w, d, c = chunk_shape

    generator = build_ceaset_generator_only(
        modality,
        chunk_shape,
        use_raw_te_film=RAW_TE_FILM_ENABLED,
    )

    with tf.device(CRITIC_DEVICE):
        critic = TEConditionedCritic()
        dummy_c = {
            "image": tf.zeros([1, h, w, d, 1], dtype=tf.float32),
            "physics_baseline": tf.zeros([1, h, w, d, 1], dtype=tf.float32),
            "target_te_ms": tf.constant([[21.5]], dtype=tf.float32),
        }
        _ = critic(dummy_c, training=False)

    return generator, critic


# During the crash-diagnosis phase execute GPU ops synchronously so failures are
# attributed to the operation that triggered them rather than a later sync.
if TRAINING_SYNC_EXECUTION_DEBUG:
    try:
        tf.config.experimental.set_synchronous_execution(True)
        print("TensorFlow synchronous execution: ON (diagnostic mode)")
    except Exception as exc:
        print("[WARN] Could not enable synchronous execution:", exc)


def _stats_for_case(case_id: str, stats: pd.DataFrame) -> pd.DataFrame:
    out = stats[stats["case_id"].astype(str) == str(case_id)].copy()
    if out.empty:
        raise KeyError(f"No CEASET statistics for {case_id}")
    return out


def _choose_input_count(n_available: int, rng: np.random.Generator) -> int:
    max_k = min(MAX_INPUT_ECHOES, n_available)
    ks = np.arange(MIN_INPUT_ECHOES, max_k + 1, dtype=int)
    weights = np.asarray(
        [INPUT_COUNT_PROBABILITIES.get(int(k), 1.0) for k in ks],
        dtype=float,
    )
    weights /= weights.sum()
    return int(rng.choice(ks, p=weights))


def _regime_for_subset(
    input_indices: Sequence[int],
    target_idx: int,
    times: np.ndarray,
) -> str:
    tin = times[list(input_indices)]
    tt = float(times[target_idx])
    if np.all(tin < tt):
        return "forward"
    if np.all(tin > tt):
        return "backward"
    if np.min(tin) < tt < np.max(tin):
        return "interpolation"
    return "other"


def sample_echo_configuration(
    echo_times_ms: Sequence[float],
    rng: np.random.Generator,
) -> tuple[int, list[int], str]:
    """
    Supervised CEASET task:
      - target is one genuinely acquired echo
      - 1-4 DIFFERENT acquired echoes are supplied as inputs
    """
    times = np.asarray(echo_times_ms, dtype=float)
    if len(times) < 2:
        raise ValueError("Training requires at least two acquired echoes")

    target_idx = int(rng.integers(0, len(times)))
    remaining = [i for i in range(len(times)) if i != target_idx]
    k = _choose_input_count(len(remaining), rng)

    combos = list(combinations(remaining, k))
    by_regime: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for combo in combos:
        by_regime[_regime_for_subset(combo, target_idx, times)].append(combo)

    regimes = [r for r, values in by_regime.items() if values]
    weights = np.asarray(
        [
            SAMPLING_REGIME_WEIGHTS.get(
                r,
                SAMPLING_REGIME_WEIGHTS["other"],
            )
            for r in regimes
        ],
        dtype=float,
    )
    weights /= weights.sum()

    regime = str(rng.choice(regimes, p=weights))
    choices = by_regime[regime]
    combo = list(choices[int(rng.integers(0, len(choices)))])

    # Permutation augmentation: echo slot must never encode TE identity.
    rng.shuffle(combo)
    return target_idx, combo, regime


def _augment_spatial_np(
    arrays: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    outputs = [np.asarray(x) for x in arrays]
    if not TRAINING_RANDOM_FLIPS:
        return outputs

    flips = [bool(rng.integers(0, 2)) for _ in range(3)]
    result = []
    for arr in outputs:
        y = arr
        # echo_images = [K,H,W,D,C], target/baseline/mask = [H,W,D,C]
        offset = 1 if arr.ndim == 5 else 0
        for spatial_axis, do_flip in enumerate(flips):
            if do_flip:
                y = np.flip(y, axis=offset + spatial_axis)
        result.append(np.ascontiguousarray(y))
    return result


def make_ceaset_sample_from_configuration(
    row: pd.Series,
    modality: str,
    stats: pd.DataFrame,
    target_idx: int,
    input_indices: Sequence[int],
    *,
    augment_rng: np.random.Generator | None = None,
    continuous_auxiliary: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    case_id = str(row["case_id"])
    echo_times = np.asarray(json.loads(row["echo_times_ms"]), dtype=np.float32)
    echo_labels = list(CFG.echo_labels)

    target_idx = int(target_idx)
    input_indices = [int(i) for i in input_indices]

    if target_idx in input_indices:
        raise ValueError("Target echo may not also be an input echo.")
    if not (MIN_INPUT_ECHOES <= len(input_indices) <= MAX_INPUT_ECHOES):
        raise ValueError(f"Expected {MIN_INPUT_ECHOES}-{MAX_INPUT_ECHOES} input echoes.")

    selected_labels = [echo_labels[i] for i in input_indices]
    selected_times = [float(echo_times[i]) for i in input_indices]
    target_label = echo_labels[target_idx]
    regime = _regime_for_subset(input_indices, target_idx, echo_times)

    case_stats = _stats_for_case(case_id, stats)
    scale = choose_reversible_scale(
        case_stats,
        selected_labels,
        modality,
    )

    mask = load_chunk(row["mask_path"]).astype(np.float32)

    selected = [
        apply_reversible_normalization(
            load_chunk(row[f"{modality}_{echo_labels[i]}_path"]),
            scale,
        ).astype(np.float32)
        for i in input_indices
    ]

    # Phase physics uses corresponding magnitude as an SNR/reliability proxy.
    # It is used ONLY to weight phi/TE frequency estimates; it is not an extra
    # phase-generator image input.
    phase_reliability_selected = None
    if (
        modality == "phase"
        and PHASE_PHYSICS_USE_MAGNITUDE_SNR_WEIGHTS
    ):
        phase_reliability_selected = [
            np.maximum(
                load_chunk(row[f"magnitude_{echo_labels[i]}_path"]),
                0.0,
            ).astype(np.float32)
            for i in input_indices
        ]

    target = apply_reversible_normalization(
        load_chunk(row[f"{modality}_{target_label}_path"]),
        scale,
    ).astype(np.float32)

    # ---------------------------------------------------------------------
    # IMPORTANT PERFORMANCE CHANGE:
    #
    # Batch size is 1, therefore K may genuinely vary from 1-4. We no longer
    # encode zero-padded dummy echoes. This can substantially reduce generator
    # work because the sampling distribution strongly favors 1-2 inputs.
    # ---------------------------------------------------------------------
    if TRAINING_DYNAMIC_ECHO_AXIS:
        echo_images = np.stack(selected, axis=0).astype(np.float32)
        echo_times_arr = np.asarray(selected_times, dtype=np.float32)
        echo_mask = np.ones((len(selected),), dtype=np.float32)
        phase_reliability_arr = (
            np.stack(phase_reliability_selected, axis=0).astype(np.float32)
            if phase_reliability_selected is not None
            else None
        )
    else:
        spatial_shape = selected[0].shape
        echo_images = np.zeros(
            (MAX_INPUT_ECHOES, *spatial_shape),
            dtype=np.float32,
        )
        echo_times_arr = np.zeros((MAX_INPUT_ECHOES,), dtype=np.float32)
        echo_mask = np.zeros((MAX_INPUT_ECHOES,), dtype=np.float32)
        for slot, (img, te) in enumerate(zip(selected, selected_times)):
            echo_images[slot] = img
            echo_times_arr[slot] = te
            echo_mask[slot] = 1.0
        if phase_reliability_selected is not None:
            phase_reliability_arr = np.zeros_like(echo_images, dtype=np.float32)
            for slot, rel in enumerate(phase_reliability_selected):
                phase_reliability_arr[slot] = rel
        else:
            phase_reliability_arr = None

    baseline = physics_baseline_np(
        modality,
        echo_images,
        echo_times_arr,
        echo_mask,
        float(echo_times[target_idx]),
        mask,
        reliability_images=phase_reliability_arr,
    )

    aux_target_te_ms = None
    aux_baseline = None
    if (
        bool(continuous_auxiliary)
        and CONTINUOUS_AUXILIARY_TRAINING
        and augment_rng is not None
    ):
        lo, hi = map(float, CONTINUOUS_TARGET_TRAIN_RANGE_MS)
        aux_target_te_ms = float(augment_rng.uniform(lo, hi))
        aux_baseline = physics_baseline_np(
            modality,
            echo_images,
            echo_times_arr,
            echo_mask,
            aux_target_te_ms,
            mask,
            reliability_images=phase_reliability_arr,
        )

    if augment_rng is not None:
        arrays = [echo_images, target, baseline, mask]
        if aux_baseline is not None:
            arrays.append(aux_baseline)
        augmented = _augment_spatial_np(arrays, augment_rng)
        echo_images, target, baseline, mask = augmented[:4]
        if aux_baseline is not None:
            aux_baseline = augmented[4]

    inputs = {
        "echo_images": echo_images.astype(np.float32),
        "echo_times_ms": echo_times_arr.astype(np.float32),
        "echo_mask": echo_mask.astype(np.float32),
        "target_te_ms": np.asarray(
            [float(echo_times[target_idx])],
            dtype=np.float32,
        ),
        "physics_baseline": baseline.astype(np.float32),
        "brain_mask": mask.astype(np.float32),
    }
    if aux_baseline is not None and aux_target_te_ms is not None:
        inputs["aux_target_te_ms"] = np.asarray(
            [aux_target_te_ms],
            dtype=np.float32,
        )
        inputs["aux_physics_baseline"] = aux_baseline.astype(np.float32)

    meta = {
        "case_id": case_id,
        "target_echo": target_label,
        "target_te_ms": float(echo_times[target_idx]),
        "input_echoes": selected_labels,
        "input_tes_ms": selected_times,
        "regime": regime,
        "scale": float(scale),
    }
    return inputs, target, meta


def make_ceaset_sample(
    row: pd.Series,
    modality: str,
    stats: pd.DataFrame,
    rng: np.random.Generator,
    *,
    augment: bool = True,
    auxiliary: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    echo_times = np.asarray(json.loads(row["echo_times_ms"]), dtype=np.float32)
    target_idx, input_indices, _ = sample_echo_configuration(echo_times, rng)
    return make_ceaset_sample_from_configuration(
        row,
        modality,
        stats,
        target_idx,
        input_indices,
        augment_rng=rng if augment else None,
        continuous_auxiliary=bool(auxiliary),
    )


def _ceaset_output_signature(chunk_shape: tuple[int, int, int, int]):
    h, w, d, c = chunk_shape
    k_dim = None if TRAINING_DYNAMIC_ECHO_AXIS else MAX_INPUT_ECHOES

    return (
        {
            "echo_images": tf.TensorSpec((k_dim, h, w, d, 1), tf.float32),
            "echo_times_ms": tf.TensorSpec((k_dim,), tf.float32),
            "echo_mask": tf.TensorSpec((k_dim,), tf.float32),
            "target_te_ms": tf.TensorSpec((1,), tf.float32),
            "physics_baseline": tf.TensorSpec((h, w, d, 1), tf.float32),
            "brain_mask": tf.TensorSpec((h, w, d, 1), tf.float32),
        },
        tf.TensorSpec((h, w, d, 1), tf.float32),
    )


def _tfdata_prefetch_value():
    return (
        tf.data.AUTOTUNE
        if int(TRAINING_PREFETCH_BATCHES) < 0
        else int(TRAINING_PREFETCH_BATCHES)
    )


def make_ceaset_dataset(
    manifest: pd.DataFrame,
    stats: pd.DataFrame,
    modality: str,
    split: str,
    epoch: int,
    batch_size: int = TRAINING_BATCH_SIZE,
    random_chunk_sample: int | None = TRAINING_RANDOM_CHUNK_SAMPLE,
    *,
    seed_override: int | None = None,
    shuffle: bool = True,
    augment: bool | None = None,
) -> tf.data.Dataset:
    """
    Parallel GPU-feeding CEASET input pipeline.

    MRI/HDF5 loading, normalization, random echo selection, physics-baseline
    calculation and augmentation are executed in parallel tf.data workers via
    tf.numpy_function. Prefetching overlaps this CPU/I/O work with GPU training.

    Every sample receives a deterministic per-sample RNG seed, so fixed
    validation remains reproducible even though training workers run in parallel.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if TRAINING_DYNAMIC_ECHO_AXIS and batch_size != 1:
        raise ValueError(
            "batch_size > 1 requires TRAINING_DYNAMIC_ECHO_AXIS=False."
        )

    subset = manifest[
        manifest["split"].astype(str) == str(split)
    ].reset_index(drop=True)

    if subset.empty:
        raise RuntimeError(f"No chunks for split={split}")

    base_seed = (
        int(seed_override)
        if seed_override is not None
        else TRAINING_SEED
        + 10007 * int(epoch)
        + (0 if modality == "magnitude" else 1000003)
    )
    order_rng = np.random.default_rng(base_seed)

    order = np.arange(len(subset), dtype=np.int64)
    if shuffle:
        order_rng.shuffle(order)

    if random_chunk_sample is not None:
        order = order[: min(len(order), int(random_chunk_sample))]

    sample_chunk = load_chunk(
        subset.iloc[int(order[0])][
            f"{modality}_{CFG.echo_labels[0]}_path"
        ]
    )
    h, w, d, _ = tuple(sample_chunk.shape)
    k_dim = None if TRAINING_DYNAMIC_ECHO_AXIS else int(MAX_INPUT_ECHOES)

    if augment is None:
        augment = str(split) == "training"
    augment = bool(augment)

    records = []
    sample_sequence = 0
    for raw_idx in order:
        for rep_idx in range(int(TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH)):
            records.append((int(raw_idx), int(rep_idx), int(sample_sequence)))
            sample_sequence += 1

    record_array = np.asarray(records, dtype=np.int64)
    if record_array.size == 0:
        raise RuntimeError(f"No sample records for split={split}")

    # Avoid filtering the complete stats DataFrame for every single sample.
    stats_by_case = {
        str(case_id): group.reset_index(drop=True)
        for case_id, group in stats.groupby(
            stats["case_id"].astype(str),
            sort=False,
        )
    }

    def _py_make_sample(record_np):
        raw_idx, rep_idx, sequence_idx = [
            int(x) for x in np.asarray(record_np).reshape(-1)
        ]
        row = subset.iloc[raw_idx]

        sample_seed = (
            int(base_seed)
            + (17 if modality == "magnitude" else 31)
            + 104729 * int(sequence_idx)
            + 1009 * int(rep_idx)
        ) % (2**63 - 1)
        local_rng = np.random.default_rng(sample_seed)

        case_id = str(row["case_id"])
        case_stats = stats_by_case.get(case_id)
        if case_stats is None or case_stats.empty:
            raise KeyError(f"No CEASET statistics for {case_id}")

        inputs, target, _ = make_ceaset_sample(
            row,
            modality,
            case_stats,
            local_rng,
            augment=augment,
        )

        return (
            inputs["echo_images"].astype(np.float32, copy=False),
            inputs["echo_times_ms"].astype(np.float32, copy=False),
            inputs["echo_mask"].astype(np.float32, copy=False),
            inputs["target_te_ms"].astype(np.float32, copy=False),
            inputs["physics_baseline"].astype(np.float32, copy=False),
            inputs["brain_mask"].astype(np.float32, copy=False),
            target.astype(np.float32, copy=False),
        )

    def _tf_make_sample(record):
        outputs = tf.numpy_function(
            _py_make_sample,
            [record],
            Tout=[
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
            ],
        )
        (
            echo_images,
            echo_times_ms,
            echo_mask,
            target_te_ms,
            physics_baseline,
            brain_mask,
            target,
        ) = outputs

        # Restore static shape information lost by numpy_function.
        echo_images.set_shape((k_dim, h, w, d, 1))
        echo_times_ms.set_shape((k_dim,))
        echo_mask.set_shape((k_dim,))
        target_te_ms.set_shape((1,))
        physics_baseline.set_shape((h, w, d, 1))
        brain_mask.set_shape((h, w, d, 1))
        target.set_shape((h, w, d, 1))

        return (
            {
                "echo_images": echo_images,
                "echo_times_ms": echo_times_ms,
                "echo_mask": echo_mask,
                "target_te_ms": target_te_ms,
                "physics_baseline": physics_baseline,
                "brain_mask": brain_mask,
            },
            target,
        )

    ds = tf.data.Dataset.from_tensor_slices(record_array)

    is_training = str(split) == "training"
    parallel_calls = max(1, int(TRAINING_IO_PARALLEL_CALLS))
    ds = ds.map(
        _tf_make_sample,
        num_parallel_calls=parallel_calls,
        deterministic=not is_training,
    )

    ds = ds.batch(
        batch_size,
        drop_remainder=bool(TRAINING_DROP_REMAINDER and is_training),
    )

    options = tf.data.Options()
    options.threading.private_threadpool_size = max(
        parallel_calls,
        int(TRAINING_IO_PRIVATE_THREADPOOL_SIZE),
    )
    options.threading.max_intra_op_parallelism = 1
    options.deterministic = not is_training
    ds = ds.with_options(options)

    prefetch_value = _tfdata_prefetch_value()
    if prefetch_value != 0:
        ds = ds.prefetch(prefetch_value)

    return ds


def expected_dataset_batches(
    manifest: pd.DataFrame,
    split: str,
    *,
    random_chunk_sample: int | None = None,
    batch_size: int = TRAINING_BATCH_SIZE,
) -> int:
    n = int((manifest["split"].astype(str) == str(split)).sum())
    if random_chunk_sample is not None:
        n = min(n, int(random_chunk_sample))
    samples = n * int(TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH)
    if bool(TRAINING_DROP_REMAINDER and str(split) == "training"):
        return int(samples // int(batch_size))
    return int(math.ceil(samples / int(batch_size)))


# =============================================================================
# DIRECT PYTHON SAMPLE ITERATOR
# =============================================================================

def iter_ceaset_samples_direct(
    manifest: pd.DataFrame,
    stats: pd.DataFrame,
    modality: str,
    split: str,
    epoch: int,
    random_chunk_sample: int | None = TRAINING_RANDOM_CHUNK_SAMPLE,
    *,
    seed_override: int | None = None,
    shuffle: bool = True,
    augment: bool | None = None,
):
    """
    Yield one CEASET batch at a time without tf.data.

    This deliberately mirrors the staged crash probe that passed successfully:
        .npy -> NumPy sample construction -> tf.convert_to_tensor -> GPU step

    There are:
      - no tf.data worker threads;
      - no tf.numpy_function callbacks;
      - no private thread pool;
      - no background prefetch queue;
      - no padded echo slots.

    Batch size is exactly 1 and the echo axis remains genuinely dynamic K=1..4.
    """
    if int(TRAINING_BATCH_SIZE) != 1:
        raise RuntimeError(
            "Direct CEASET iterator currently requires TRAINING_BATCH_SIZE=1."
        )
    if not TRAINING_DYNAMIC_ECHO_AXIS:
        raise RuntimeError(
            "Direct CEASET iterator expects TRAINING_DYNAMIC_ECHO_AXIS=True."
        )

    subset = manifest[
        manifest["split"].astype(str) == str(split)
    ].reset_index(drop=True)

    if subset.empty:
        raise RuntimeError(f"No chunks for split={split}")

    base_seed = (
        int(seed_override)
        if seed_override is not None
        else TRAINING_SEED
        + 10007 * int(epoch)
        + (0 if modality == "magnitude" else 1000003)
    )

    order_rng = np.random.default_rng(base_seed)
    order = np.arange(len(subset), dtype=np.int64)

    if shuffle:
        order_rng.shuffle(order)

    if random_chunk_sample is not None:
        order = order[: min(len(order), int(random_chunk_sample))]

    if augment is None:
        augment = str(split) == "training"
    augment = bool(augment)

    stats_by_case = {
        str(case_id): group.reset_index(drop=True)
        for case_id, group in stats.groupby(
            stats["case_id"].astype(str),
            sort=False,
        )
    }

    sample_sequence = 0

    for raw_idx in order:
        row = subset.iloc[int(raw_idx)]
        case_id = str(row["case_id"])

        case_stats = stats_by_case.get(case_id)
        if case_stats is None or case_stats.empty:
            raise KeyError(f"No CEASET statistics for {case_id}")

        for rep_idx in range(int(TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH)):
            sample_seed = (
                int(base_seed)
                + (17 if modality == "magnitude" else 31)
                + 104729 * int(sample_sequence)
                + 1009 * int(rep_idx)
            ) % (2**63 - 1)

            local_rng = np.random.default_rng(sample_seed)

            use_auxiliary = (
                augment
                and CONTINUOUS_AUXILIARY_TRAINING
                and int(CONTINUOUS_AUXILIARY_EVERY_N_BATCHES) > 0
                and ((int(sample_sequence) + 1) % int(CONTINUOUS_AUXILIARY_EVERY_N_BATCHES) == 0)
            )

            inputs_np, target_np, meta = make_ceaset_sample(
                row,
                modality,
                case_stats,
                local_rng,
                augment=augment,
                auxiliary=use_auxiliary,
            )

            # Batch dimension only. The echo dimension itself is NOT padded.
            inputs_tf = {
                key: tf.convert_to_tensor(
                    value[None, ...],
                    dtype=tf.float32,
                )
                for key, value in inputs_np.items()
            }

            target_tf = tf.convert_to_tensor(
                target_np[None, ...],
                dtype=tf.float32,
            )

            yield inputs_tf, target_tf, meta

            # Explicitly drop host references before constructing the next sample.
            del inputs_np, target_np, inputs_tf, target_tf
            sample_sequence += 1


def expected_direct_samples(
    manifest: pd.DataFrame,
    split: str,
    *,
    random_chunk_sample: int | None = None,
) -> int:
    n = int(
        (manifest["split"].astype(str) == str(split)).sum()
    )
    if random_chunk_sample is not None:
        n = min(n, int(random_chunk_sample))

    return int(
        n * int(TRAINING_SAMPLES_PER_CHUNK_PER_EPOCH)
    )


def evaluate_ceaset_direct(
    generator,
    manifest: pd.DataFrame,
    stats: pd.DataFrame,
    modality: str,
    *,
    max_chunks: int | None = EVAL_MAX_CHUNKS,
    seed: int = FIXED_EVAL_SEED,
    shuffle: bool = FIXED_EVAL_SHUFFLE,
    log_every_batches: int = 25,
    label: str = "EVAL",
):
    """
    Fixed validation using the same direct, non-tf.data path as training.
    """
    metrics = []
    start = time.time()

    iterator = iter_ceaset_samples_direct(
        manifest,
        stats,
        modality,
        "eval",
        epoch=0,
        random_chunk_sample=max_chunks,
        seed_override=int(seed),
        shuffle=bool(shuffle),
        augment=False,
    )

    for batch_idx, (inputs, real, _meta) in enumerate(iterator, start=1):
        check_training_stop()
        fake = generator(inputs, training=False)

        # Explicit synchronization occurs here.
        mask = inputs["brain_mask"]
        mae = float(masked_l1(real, fake, mask).numpy())
        mse = float(
            masked_mean(
                tf.square(real - fake),
                mask,
            ).numpy()
        )

        metrics.append((mae, mse))

        if log_every_batches and batch_idx % int(log_every_batches) == 0:
            arr = np.asarray(metrics, dtype=float)
            print(
                f"[{label}] batch={batch_idx:04d} "
                f"elapsed={(time.time()-start)/60:.1f}min "
                f"MAE={arr[:,0].mean():.5f} "
                f"RMSE={np.sqrt(arr[:,1].mean()):.5f} "
                f"{_gpu_memory_string()}",
                flush=True,
            )

        del inputs, real, fake

    if not metrics:
        return {"mae": np.nan, "rmse": np.nan}

    arr = np.asarray(metrics, dtype=float)
    return {
        "mae": float(arr[:, 0].mean()),
        "rmse": float(np.sqrt(arr[:, 1].mean())),
    }


def masked_mean(x, mask, eps=1e-6):
    mask = tf.cast(mask, x.dtype)
    return tf.reduce_sum(x * mask) / (tf.reduce_sum(mask) + tf.cast(eps, x.dtype))


def masked_l1(y_true, y_pred, mask):
    return masked_mean(tf.abs(y_true-y_pred), mask)


def gradient3d(x):
    dx = tf.pad(x[:,1:]-x[:,:-1], [[0,0],[0,1],[0,0],[0,0],[0,0]])
    dy = tf.pad(x[:,:,1:]-x[:,:,:-1], [[0,0],[0,0],[0,1],[0,0],[0,0]])
    dz = tf.pad(x[:,:,:,1:]-x[:,:,:,:-1], [[0,0],[0,0],[0,0],[0,1],[0,0]])
    return dx,dy,dz


def masked_edge_l1(y_true, y_pred, mask):
    tg=gradient3d(y_true); pg=gradient3d(y_pred)
    return sum(masked_mean(tf.abs(a-b), mask) for a,b in zip(tg,pg))/3.0


def _gaussian_kernel_3d_np(sigmas):
    kernels = []
    for sigma in sigmas:
        sigma = float(sigma)
        if sigma <= 0:
            kernels.append(np.asarray([1.0], dtype=np.float32))
            continue
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (coords / sigma) ** 2)
        k /= np.sum(k)
        kernels.append(k.astype(np.float32))
    kernel = (
        kernels[0][:, None, None]
        * kernels[1][None, :, None]
        * kernels[2][None, None, :]
    )
    kernel /= np.sum(kernel)
    return kernel[..., None, None].astype(np.float32)


_PHYSICS_GAUSSIAN_KERNEL = tf.constant(
    _gaussian_kernel_3d_np(PHYSICS_GAUSSIAN_SIGMA_VOXELS),
    dtype=tf.float32,
)


def masked_anisotropic_gaussian_blur3d(x, mask, eps=1e-6):
    """
    Mask-normalized Gaussian blur.  Sigma=(1.5,1.5,0) means only in-plane
    neighbors contribute; adjacent axial slices are deliberately not mixed.
    """
    x = tf.cast(x, tf.float32)
    mask = tf.cast(mask, tf.float32)
    numerator = tf.nn.conv3d(
        x * mask,
        _PHYSICS_GAUSSIAN_KERNEL,
        strides=[1, 1, 1, 1, 1],
        padding="SAME",
    )
    denominator = tf.nn.conv3d(
        mask,
        _PHYSICS_GAUSSIAN_KERNEL,
        strides=[1, 1, 1, 1, 1],
        padding="SAME",
    )
    blurred = numerator / (denominator + tf.cast(eps, tf.float32))
    return blurred, tf.clip_by_value(denominator, 0.0, 1.0)


def masked_huber(y_true, y_pred, mask, delta=PHYSICS_HUBER_DELTA):
    diff = tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))
    delta = tf.cast(delta, tf.float32)
    quadratic = tf.minimum(diff, delta)
    linear = diff - quadratic
    # Scaled Huber: has approximately L1 units for easier weighting.
    value = 0.5 * tf.square(quadratic) / (delta + 1e-8) + linear
    return masked_mean(value, mask)


def masked_lncc_loss(
    a,
    b,
    mask,
    window=PHYSICS_LNCC_WINDOW_VOXELS,
    eps=1e-6,
):
    """
    Local normalized cross-correlation on an anisotropic x-y neighborhood.
    Unlike global NCC, this checks whether local structures change together.
    The separate Huber term anchors absolute coarse intensity/phase.
    """
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    mask = tf.cast(mask, tf.float32)
    wx, wy, wz = [int(v) for v in window]
    ksize = [1, wx, wy, wz, 1]
    strides = [1, 1, 1, 1, 1]

    pool = lambda z: tf.nn.avg_pool3d(
        z,
        ksize=ksize,
        strides=strides,
        padding="SAME",
    )

    w = pool(mask)
    denom = w + tf.cast(eps, tf.float32)
    mean_a = pool(a * mask) / denom
    mean_b = pool(b * mask) / denom
    mean_aa = pool(tf.square(a) * mask) / denom
    mean_bb = pool(tf.square(b) * mask) / denom
    mean_ab = pool(a * b * mask) / denom

    var_a = tf.maximum(mean_aa - tf.square(mean_a), 0.0)
    var_b = tf.maximum(mean_bb - tf.square(mean_b), 0.0)
    cov = mean_ab - mean_a * mean_b

    corr = cov / tf.sqrt(var_a * var_b + tf.cast(eps, tf.float32))
    corr = tf.clip_by_value(corr, -1.0, 1.0)

    valid = tf.cast(
        (w >= float(PHYSICS_LNCC_MIN_MASK_FRACTION))
        & (var_a >= float(PHYSICS_LNCC_MIN_VARIANCE))
        & (var_b >= float(PHYSICS_LNCC_MIN_VARIANCE)),
        tf.float32,
    )
    # 1-corr (not corr^2): inverted local structure is penalized, not rewarded.
    return masked_mean(1.0 - corr, valid)


def coarse_structural_physics_loss(prediction, baseline, mask):
    pred_low, support = masked_anisotropic_gaussian_blur3d(
        prediction,
        mask,
    )
    base_low, _ = masked_anisotropic_gaussian_blur3d(
        baseline,
        mask,
    )
    coarse_mask = tf.cast(mask, tf.float32) * tf.cast(
        support >= float(PHYSICS_LNCC_MIN_MASK_FRACTION),
        tf.float32,
    )

    intensity = masked_huber(
        base_low,
        pred_low,
        coarse_mask,
        delta=PHYSICS_HUBER_DELTA,
    )
    structure = masked_lncc_loss(
        base_low,
        pred_low,
        coarse_mask,
        window=PHYSICS_LNCC_WINDOW_VOXELS,
    )
    total = (
        tf.cast(PHYSICS_INTENSITY_FRACTION, tf.float32) * intensity
        + tf.cast(PHYSICS_STRUCTURE_FRACTION, tf.float32) * structure
    )
    return total, intensity, structure


def masked_standardize_for_perceptual(x, mask):
    mask = tf.cast(mask, x.dtype)
    count = tf.reduce_sum(mask, axis=[1,2,3,4], keepdims=True) + 1e-6
    mean = tf.reduce_sum(x*mask, axis=[1,2,3,4], keepdims=True)/count
    var = tf.reduce_sum(tf.square(x-mean)*mask, axis=[1,2,3,4], keepdims=True)/count
    z=(x-mean)/tf.sqrt(var+1e-6)
    z=tf.clip_by_value(z/3.0, -1.0, 1.0)
    return z*mask


class SlicePerceptualExtractor:
    def __init__(self, target_shape, img_size=128, slice_stride=2):
        self.enabled = RADIMAGENET_RESNET50_WEIGHTS.is_file()
        self.model = None
        if not self.enabled:
            print(f"[WARN] RadImageNet weights not found; perceptual loss will be zero: {RADIMAGENET_RESNET50_WEIGHTS}")
            return
        base = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(img_size,img_size,3))
        base.load_weights(str(RADIMAGENET_RESNET50_WEIGHTS))
        trunk = tf.keras.Model(base.input, base.get_layer("conv1_relu").output)
        trunk.trainable=False
        h,w,d,_ = target_shape
        sel=list(range(0,d,slice_stride)); deff=len(sel)
        inp=tf.keras.Input(shape=target_shape)
        x=layers.Lambda(lambda t: tf.transpose(t,[0,3,1,2,4]))(inp)
        x=layers.Lambda(lambda t: tf.gather(t,sel,axis=1))(x)
        x=layers.Lambda(lambda t: tf.tile(t,[1,1,1,1,3]))(x)
        x=layers.Lambda(lambda t: tf.reshape(t,[tf.shape(t)[0]*deff,h,w,3]))(x)
        x=layers.Resizing(img_size,img_size)(x)
        x=layers.Lambda(lambda u:(u+1.0)*127.5)(x)
        x=trunk(x)
        ho,wo,co=trunk.output_shape[1:]
        x=layers.Lambda(lambda t: tf.reshape(t,[tf.shape(t)[0]//deff,deff,ho,wo,co]))(x)
        x=layers.Lambda(lambda t: tf.reduce_mean(t,axis=1))(x)
        self.model=tf.keras.Model(inp,x,name="ceaset_radimagenet_perceptual")
        self.model.trainable=False

    def loss(self, y_true, y_pred, mask):
        if not self.enabled or self.model is None:
            return tf.constant(0.0, tf.float32)
        a=masked_standardize_for_perceptual(y_true,mask)
        b=masked_standardize_for_perceptual(y_pred,mask)
        fa=tf.stop_gradient(self.model(a,training=False)); fb=self.model(b,training=False)
        fa=tf.nn.l2_normalize(fa,axis=-1); fb=tf.nn.l2_normalize(fb,axis=-1)
        return tf.reduce_mean(tf.abs(fa-fb))


# Count trainer/model constructions in this Python process.
CEASET_TRAINER_INSTANCES_CREATED = 0
CEASET_PRODUCTION_TRAINERS_COMPLETED = 0

class CEASETTrainer:
    """
    Memory-isolated dual-GPU WGAN-GP trainer.

    Generator + generator optimizer:
        logical GPU:0

    Critic + WGAN gradient penalty + critic optimizer:
        logical GPU:1

    The critic update builds the generator fake WITHOUT a generator tape, then
    transfers only the final fake volume to GPU:1. The expensive second-order
    GP derivative therefore never shares the generator's ~18 GiB activation
    allocator.

    During the generator update, TensorFlow transports the generated candidate
    to GPU:1 for critic scoring and automatically propagates the adversarial
    gradient back across the device boundary into the generator on GPU:0.
    """

    def __init__(self, modality: str, chunk_shape: tuple[int, int, int, int]):
        global CEASET_TRAINER_INSTANCES_CREATED
        CEASET_TRAINER_INSTANCES_CREATED += 1
        self.trainer_instance_number = CEASET_TRAINER_INSTANCES_CREATED
        print(
            f"[TRAINER] Constructing dual-GPU WGAN-GP trainer "
            f"#{self.trainer_instance_number}",
            flush=True,
        )

        self.modality = modality
        self.chunk_shape = tuple(chunk_shape)
        self.hp = TRAINING_HYPERPARAMETERS[modality]
        self.generator_device = GENERATOR_DEVICE
        self.critic_device = CRITIC_DEVICE

        self.generator, self.critic = build_ceaset_models(modality, chunk_shape)

        with tf.device(self.generator_device):
            self.g_optimizer = tf.keras.optimizers.Adam(
                learning_rate=float(self.hp["generator_lr"]),
                beta_1=0.0,
                beta_2=0.9,
                name="generator_adam",
            )
            self.g_optimizer.build(self.generator.trainable_variables)

            if TRAINING_ENABLE_PERCEPTUAL_LOSS and float(self.hp["perc_weight"]) > 0:
                self.perceptual = SlicePerceptualExtractor(chunk_shape)
            else:
                self.perceptual = None

        with tf.device(self.critic_device):
            # Adam with beta1=0, beta2=0.9 is the conventional WGAN-GP choice.
            self.d_optimizer = tf.keras.optimizers.Adam(
                learning_rate=float(self.hp["critic_lr"]),
                beta_1=0.0,
                beta_2=0.9,
                epsilon=1e-7,
                name="critic_adam_gp",
            )
            self.d_optimizer.build(self.critic.trainable_variables)

        # Separate state namespace: never mix clipped-WGAN optimizer/critic
        # state with the new WGAN-GP formulation.
        self.state_epoch = tf.Variable(
            0, dtype=tf.int64, trainable=False, name="epoch"
        )
        self.state_root = (
            TRAINING_MODEL_ROOT
            / self.modality
            / f"training_state_{TRAINING_VARIANT_TAG}"
        )
        self.state_root.mkdir(parents=True, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(
            epoch=self.state_epoch,
            generator=self.generator,
            critic=self.critic,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            str(self.state_root),
            max_to_keep=5,
        )

        if TRAINING_COMPILE_STEPS:
            self._train_batch = tf.function(
                self._train_batch_impl,
                reduce_retracing=True,
            )
        else:
            self._train_batch = self._train_batch_impl

    def _critic_inputs(self, image, baseline, target_te):
        return {
            "image": image,
            "physics_baseline": baseline,
            "target_te_ms": target_te,
        }

    def _critic_update(self, inputs, real, preview_axis=-1):
        """
        One full-resolution WGAN-GP critic update on GPU:1.

        Important memory detail:
          - generator forward is outside the critic GradientTape;
          - only fake output is transferred to GPU:1;
          - GP inner tape watches ONLY the interpolated image;
          - baseline and TE are treated as conditioning constants.
        """
        # Generator inference on GPU:0; no generator tape or retained backward graph.
        with tf.device(self.generator_device):
            fake_g = tf.stop_gradient(
                self.generator(inputs, training=False)
            )

        with tf.device(self.critic_device):
            mask = tf.cast(inputs["brain_mask"], tf.float32)
            target_te = tf.stop_gradient(
                tf.cast(inputs["target_te_ms"], tf.float32)
            )
            baseline = tf.stop_gradient(
                tf.cast(inputs["physics_baseline"], tf.float32) * mask
            )

            real_m = tf.stop_gradient(tf.cast(real, tf.float32) * mask)
            # tf.identity under GPU:1 explicitly creates the cross-device edge.
            fake_m = tf.stop_gradient(tf.identity(fake_g)) * mask

            # Slice the exact masked fake passed to the critic. No new generator call.
            preview = None
            if preview_axis >= 0:
                volume_shape = tf.shape(fake_m)[1:4]
                middle = volume_shape[preview_axis] // 2
                def plane(volume):
                    part = tf.gather(tf.stop_gradient(volume[0, ..., 0]), middle, axis=preview_axis)
                    stride = tf.maximum((tf.shape(part) + 255) // 256, 1)
                    return part[::stride[0], ::stride[1]]
                # The generator sampled K=1..4 real input echoes in this batch.
                # Gather only their central slices on the generator device so
                # a multi-echo preview does not transfer full 3-D stacks to
                # the critic device. Respect the echo mask for padded inputs.
                with tf.device(self.generator_device):
                    input_volumes = tf.stop_gradient(inputs['echo_images'][0, ..., 0])
                    input_slices = tf.gather(input_volumes, middle, axis=preview_axis+1)
                    stride = tf.maximum((tf.shape(input_slices)[1:] + 255) // 256, 1)
                    input_slices = input_slices[:, ::stride[0], ::stride[1]]
                    input_slices = tf.boolean_mask(
                        input_slices, inputs['echo_mask'][0] > .5, axis=0)
                mask_slice = plane(mask)
                preview = tf.concat([
                    input_slices * mask_slice[None, ...],
                    tf.stack([plane(fake_m), plane(real_m), mask_slice], axis=0),
                ], axis=0)

            # Full-resolution GP by default. A factor of 2 is available only
            # as an emergency memory fallback and leaves real/fake scoring at
            # full resolution.
            gp_factor = int(WGAN_GP_DOWNSAMPLE_FACTOR)
            if gp_factor == 1:
                real_gp = real_m
                fake_gp = fake_m
                baseline_gp = baseline
            elif gp_factor > 1:
                k = int(gp_factor)
                real_gp = tf.nn.avg_pool3d(
                    real_m,
                    ksize=[1, k, k, k, 1],
                    strides=[1, k, k, k, 1],
                    padding="SAME",
                )
                fake_gp = tf.nn.avg_pool3d(
                    fake_m,
                    ksize=[1, k, k, k, 1],
                    strides=[1, k, k, k, 1],
                    padding="SAME",
                )
                baseline_gp = tf.nn.avg_pool3d(
                    baseline,
                    ksize=[1, k, k, k, 1],
                    strides=[1, k, k, k, 1],
                    padding="SAME",
                )
            else:
                raise ValueError("WGAN_GP_DOWNSAMPLE_FACTOR must be >= 1")

            batch_size = tf.shape(real_gp)[0]
            alpha = tf.random.uniform(
                [batch_size, 1, 1, 1, 1],
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
            interpolated = real_gp + alpha * (fake_gp - real_gp)

            with tf.GradientTape() as d_tape:
                real_score = self.critic(
                    self._critic_inputs(real_m, baseline, target_te),
                    training=True,
                )
                fake_score = self.critic(
                    self._critic_inputs(fake_m, baseline, target_te),
                    training=True,
                )

                # Inner tape records only dD/dx. Because this operation is
                # executed inside d_tape, d_tape can differentiate the GP back
                # into critic parameters (the required second-order derivative).
                with tf.GradientTape(
                    watch_accessed_variables=False
                ) as gp_tape:
                    gp_tape.watch(interpolated)
                    interp_score = self.critic(
                        self._critic_inputs(
                            interpolated,
                            baseline_gp,
                            target_te,
                        ),
                        training=True,
                    )
                    interp_score_sum = tf.reduce_sum(interp_score)

                interp_grad = gp_tape.gradient(
                    interp_score_sum,
                    interpolated,
                )
                grad_norm = tf.sqrt(
                    tf.reduce_sum(
                        tf.square(interp_grad),
                        axis=[1, 2, 3, 4],
                    )
                    + tf.cast(WGAN_GP_EPSILON, tf.float32)
                )
                gp = tf.reduce_mean(
                    tf.square(grad_norm - 1.0)
                )

                real_mean = tf.reduce_mean(real_score)
                fake_mean = tf.reduce_mean(fake_score)

                # Critic minimizes fake-real + lambda*GP.
                wasserstein = fake_mean - real_mean
                gap = real_mean - fake_mean
                gp_weighted = (
                    tf.cast(WGAN_GP_LAMBDA, tf.float32) * gp
                )
                d_loss = wasserstein + gp_weighted

            grads = d_tape.gradient(
                d_loss,
                self.critic.trainable_variables,
            )
            self.d_optimizer.apply_gradients(
                [
                    (g, v)
                    for g, v in zip(
                        grads,
                        self.critic.trainable_variables,
                    )
                    if g is not None
                ]
            )

            result = {
                "d_loss": d_loss,
                "d_wasserstein": wasserstein,
                "d_gap": gap,
                "gp": gp,
                "gp_weighted": gp_weighted,
                "d_real_score": real_mean,
                "d_fake_score": fake_mean,
                "d_grad_norm": tf.reduce_mean(grad_norm),
            }
            if preview is not None:
                result['_preview'] = preview
            return result

    def _generator_update(self, inputs, real):
        """
        Generator update on GPU:0 with critic scoring on GPU:1.

        The adversarial loss is written as:
            stop_gradient(mean(D(real))) - mean(D(fake))
        rather than simply -mean(D(fake)).

        These have IDENTICAL gradients with respect to the generator, but the
        gap form is invariant to an arbitrary constant offset in critic scores.
        It therefore makes the logged adversarial term interpretable.
        """
        # Reference real score is constant for the generator update.
        with tf.device(self.critic_device):
            mask_c = tf.cast(inputs["brain_mask"], tf.float32)
            baseline_c = tf.stop_gradient(
                tf.cast(inputs["physics_baseline"], tf.float32) * mask_c
            )
            target_te_c = tf.stop_gradient(
                tf.cast(inputs["target_te_ms"], tf.float32)
            )
            real_c = tf.stop_gradient(
                tf.cast(real, tf.float32) * mask_c
            )
            real_score_ref = tf.stop_gradient(
                tf.reduce_mean(
                    self.critic(
                        self._critic_inputs(
                            real_c,
                            baseline_c,
                            target_te_c,
                        ),
                        training=False,
                    )
                )
            )

        with tf.device(self.generator_device):
            mask_g = tf.cast(inputs["brain_mask"], tf.float32)

            with tf.GradientTape() as tape:
                fake, aux = self.generator(
                    inputs,
                    training=True,
                    return_aux=True,
                )

                rec = masked_l1(real, fake, mask_g)

                if (
                    TRAINING_ENABLE_EDGE_LOSS
                    and float(self.hp["edge_weight"]) > 0
                ):
                    edge = masked_edge_l1(real, fake, mask_g)
                else:
                    edge = tf.constant(0.0, tf.float32)

                if (
                    TRAINING_ENABLE_PERCEPTUAL_LOSS
                    and self.perceptual is not None
                    and float(self.hp["perc_weight"]) > 0
                ):
                    perc = self.perceptual.loss(real, fake, mask_g)
                else:
                    perc = tf.constant(0.0, tf.float32)

                if (
                    TRAINING_ENABLE_COARSE_PHYSICS_LOSS
                    and float(self.hp["physics_residual_weight"]) > 0
                ):
                    phys, phys_intensity, phys_structure = (
                        coarse_structural_physics_loss(
                            fake,
                            inputs["physics_baseline"],
                            mask_g,
                        )
                    )
                else:
                    phys = tf.constant(0.0, tf.float32)
                    phys_intensity = tf.constant(0.0, tf.float32)
                    phys_structure = tf.constant(0.0, tf.float32)

                # Critic forward lives on GPU:1, but remains inside the generator
                # tape. TensorFlow differentiates through the cross-device copy.
                with tf.device(self.critic_device):
                    fake_c = tf.identity(
                        tf.cast(fake, tf.float32) * mask_g
                    )
                    fake_score = self.critic(
                        self._critic_inputs(
                            fake_c,
                            baseline_c,
                            target_te_c,
                        ),
                        training=False,
                    )
                    fake_mean = tf.reduce_mean(fake_score)
                    adv = (
                        tf.cast(real_score_ref, tf.float32)
                        - fake_mean
                    )

                adv_weighted = (
                    tf.cast(self.hp["adv_weight"], tf.float32) * adv
                )
                rec_weighted = (
                    tf.cast(self.hp["rec_weight"], tf.float32) * rec
                )
                edge_weighted = (
                    tf.cast(self.hp["edge_weight"], tf.float32) * edge
                )
                perc_weighted = (
                    tf.cast(self.hp["perc_weight"], tf.float32) * perc
                )
                phys_weighted = (
                    tf.cast(
                        self.hp["physics_residual_weight"],
                        tf.float32,
                    ) * phys
                )

                total = (
                    adv_weighted
                    + rec_weighted
                    + edge_weighted
                    + perc_weighted
                    + phys_weighted
                )

                if self.generator.losses:
                    total = total + tf.add_n(
                        self.generator.losses
                    )

            grads = tape.gradient(
                total,
                self.generator.trainable_variables,
            )
            self.g_optimizer.apply_gradients(
                [
                    (g, v)
                    for g, v in zip(
                        grads,
                        self.generator.trainable_variables,
                    )
                    if g is not None
                ]
            )

            return {
                "g_total": total,
                "g_adv": adv,
                "g_adv_weighted": adv_weighted,
                "g_rec": rec,
                "g_rec_weighted": rec_weighted,
                "g_edge": edge,
                "g_edge_weighted": edge_weighted,
                "g_perc": perc,
                "g_perc_weighted": perc_weighted,
                "g_phys": phys,
                "g_phys_intensity": phys_intensity,
                "g_phys_structure": phys_structure,
                "g_phys_weighted": phys_weighted,
                "g_fake_score": fake_mean,
                "g_real_score_ref": real_score_ref,
            }

    def _continuous_auxiliary_update(self, inputs):
        """
        One extra generator-only update for an unmeasured TE.

        No fake target is invented.  The only objective is low-frequency /
        local-structural consistency with the deterministic physics baseline.
        This is intentionally weak and sparse, and is what extends training
        exposure from the measured ABRIM echoes out to 40 ms.
        """
        if (
            not CONTINUOUS_AUXILIARY_TRAINING
            or "aux_target_te_ms" not in inputs
            or "aux_physics_baseline" not in inputs
        ):
            zero = tf.constant(0.0, tf.float32)
            return {
                "g_aux_phys": zero,
                "g_aux_phys_intensity": zero,
                "g_aux_phys_structure": zero,
                "g_aux_phys_weighted": zero,
            }

        aux_weight = float(
            CONTINUOUS_AUXILIARY_WEIGHT_BY_MODALITY[self.modality]
        )
        with tf.device(self.generator_device):
            aux_inputs = dict(inputs)
            aux_inputs["target_te_ms"] = inputs["aux_target_te_ms"]
            aux_inputs["physics_baseline"] = inputs["aux_physics_baseline"]
            mask = tf.cast(inputs["brain_mask"], tf.float32)

            with tf.GradientTape() as tape:
                pred = self.generator(aux_inputs, training=True)
                phys, phys_intensity, phys_structure = (
                    coarse_structural_physics_loss(
                        pred,
                        inputs["aux_physics_baseline"],
                        mask,
                    )
                )
                weighted = tf.cast(aux_weight, tf.float32) * phys

            grads = tape.gradient(
                weighted,
                self.generator.trainable_variables,
            )
            self.g_optimizer.apply_gradients(
                [
                    (g, v)
                    for g, v in zip(
                        grads,
                        self.generator.trainable_variables,
                    )
                    if g is not None
                ]
            )

        return {
            "g_aux_phys": phys,
            "g_aux_phys_intensity": phys_intensity,
            "g_aux_phys_structure": phys_structure,
            "g_aux_phys_weighted": weighted,
        }

    def _train_batch_impl(self, inputs, real, critic_updates=1, preview_axis=-1):
        d_acc = None
        critic_updates = int(critic_updates)

        preview = None
        for critic_index in range(critic_updates):
            d = self._critic_update(inputs, real, preview_axis if critic_index == 0 else -1)
            if '_preview' in d:
                preview = d.pop('_preview')
            if d_acc is None:
                d_acc = d
            else:
                d_acc = {
                    key: d_acc[key] + d[key]
                    for key in d_acc
                }

        if critic_updates > 1:
            d_acc = {
                key: value / float(critic_updates)
                for key, value in d_acc.items()
            }

        g = self._generator_update(inputs, real)
        g_aux = self._continuous_auxiliary_update(inputs)

        out = {}
        out.update(d_acc)
        out.update(g)
        out.update(g_aux)
        if preview is not None:
            out['_preview'] = preview
        return out

    def train_batch(self, inputs, real, critic_updates: int = 1, preview_axis: int = -1):
        if int(critic_updates) < 1:
            raise ValueError("critic_updates must be >= 1")
        return self._train_batch(
            inputs,
            real,
            int(critic_updates),
            int(preview_axis),
        )

    def critic_updates_for_epoch(self, epoch: int) -> int:
        return int(CRITIC_UPDATES_BY_MODALITY[self.modality])

    def generator_checkpoint_path(self, epoch: int) -> Path:
        root = TRAINING_MODEL_ROOT / self.modality
        root.mkdir(parents=True, exist_ok=True)
        return root / f"generator_{RUN_NAME}_epoch{epoch:03d}.weights.h5"

    def critic_checkpoint_path(self, epoch: int) -> Path:
        root = TRAINING_MODEL_ROOT / self.modality
        root.mkdir(parents=True, exist_ok=True)
        return root / (
            f"critic_{TRAINING_VARIANT_TAG}_epoch{epoch:03d}.weights.h5"
        )

    def latest_generator_checkpoint_epoch(self) -> int:
        root = TRAINING_MODEL_ROOT / self.modality
        epochs = []
        if root.is_dir():
            for p in root.glob(f"generator_{RUN_NAME}_epoch*.weights.h5"):
                match = re.search(r"epoch(\d+)", p.name)
                if match:
                    epochs.append(int(match.group(1)))
        return max(epochs) if epochs else 0

    def maybe_resume(self) -> tuple[int, str]:
        """
        Resume priority:
          1) optimizer-inclusive checkpoint from THIS dual-GPU WGAN-GP variant;
          2) latest generator epoch weights, with critic and optimizers reset.
        """
        if not TRAINING_RESUME:
            return 0, "fresh"

        if self.checkpoint_manager.latest_checkpoint:
            status = self.checkpoint.restore(
                self.checkpoint_manager.latest_checkpoint
            )
            status.expect_partial()
            epoch = int(self.state_epoch.numpy())
            print(
                f"[RESUME] {self.modality}: full {TRAINING_GAN_MODE} state "
                f"from epoch {epoch}: "
                f"{self.checkpoint_manager.latest_checkpoint}"
            )
            return epoch, "full_state_wgan_gp_dualgpu"

        epoch = self.latest_generator_checkpoint_epoch()
        if epoch > 0:
            gp_path = self.generator_checkpoint_path(epoch)
            migration_mode = load_generator_weights_v7_compatible(
                self.generator,
                gp_path,
                self.modality,
                self.chunk_shape,
            )

            print(
                f"[RESUME] {self.modality}: generator weights from epoch "
                f"{epoch}: {gp_path}\n"
                "         Critic RESET intentionally for dual-GPU WGAN-GP.\n"
                "         Generator/critic optimizer moments are fresh. "
                "This is a controlled V7 architecture/objective transition.\n"
                f"         Generator load mode: {migration_mode}"
            )
            return epoch, f"generator_only_v7_transition:{migration_mode}"

        return 0, "fresh"

    def save_checkpoint(self, epoch: int) -> tuple[Path, Path, str]:
        gp_path = self.generator_checkpoint_path(epoch)
        dp_path = self.critic_checkpoint_path(epoch)

        self.generator.save_weights(gp_path)
        self.critic.save_weights(dp_path)

        self.state_epoch.assign(int(epoch))
        state_path = self.checkpoint_manager.save(
            checkpoint_number=int(epoch)
        )
        return gp_path, dp_path, state_path


def _gpu_memory_string() -> str:
    parts = []
    gib = 1024 ** 3
    for idx, label in ((0, "G"), (1, "D/GP")):
        try:
            info = tf.config.experimental.get_memory_info(f"GPU:{idx}")
            parts.append(
                f"{label}={info['current']/gib:.1f}GiB "
                f"peak={info['peak']/gib:.1f}GiB"
            )
        except Exception:
            parts.append(f"{label}=n/a")
    return " | ".join(parts)


def evaluate_dataset_basic(
    generator,
    dataset,
    max_batches=None,
    *,
    log_every_batches: int = 25,
    label: str = "EVAL",
):
    metrics = []
    start = time.time()

    for batch_idx, (inputs, real) in enumerate(dataset, start=1):
        check_training_stop()
        if max_batches is not None and batch_idx > int(max_batches):
            break

        fake = generator(inputs, training=False)
        mask = inputs["brain_mask"]
        mae = float(masked_l1(real, fake, mask).numpy())
        mse = float(masked_mean(tf.square(real - fake), mask).numpy())
        metrics.append((mae, mse))

        if log_every_batches and batch_idx % int(log_every_batches) == 0:
            arr = np.asarray(metrics, dtype=float)
            print(
                f"[{label}] batch={batch_idx:04d} "
                f"elapsed={(time.time()-start)/60:.1f}min "
                f"MAE={arr[:,0].mean():.5f} "
                f"RMSE={np.sqrt(arr[:,1].mean()):.5f}",
                flush=True,
            )

    if not metrics:
        return {"mae": np.nan, "rmse": np.nan}

    arr = np.asarray(metrics, dtype=float)
    return {
        "mae": float(arr[:, 0].mean()),
        "rmse": float(np.sqrt(arr[:, 1].mean())),
    }


def train_ceaset_modality(
    modality: str,
    epochs: int = TRAINING_EPOCHS,
    *,
    log_every_batches: int = TRAINING_LOG_EVERY_BATCHES,
    log_every_minutes: float = TRAINING_LOG_EVERY_MINUTES,
) -> pd.DataFrame:
    """
    Train one CEASET modality using two GPUs (generator / critic+GP).

    This is manual model parallelism, not MirroredStrategy. Both GPUs are
    dedicated to one modality during training.
    """
    assert_direct_loop_runtime_config()

    # A legitimate completed production trainer may be followed by the other
    # modality.  A probe/benchmark-created trainer still blocks production.
    created = int(globals().get("CEASET_TRAINER_INSTANCES_CREATED", 0))
    completed = int(globals().get("CEASET_PRODUCTION_TRAINERS_COMPLETED", 0))
    if created != completed:
        raise RuntimeError(
            "Production CEASET training detected a trainer that was constructed "
            "outside a completed production run (typically a probe/benchmark or "
            "a failed prior trainer). Restart the kernel and run from the top. "
            f"created={created}, completed={completed}"
        )

    # Release the previous modality before checking physical free memory.
    tf.keras.backend.clear_session()
    gc.collect()

    assert_training_hardware()
    validate_training_artifacts(CFG)

    tf.keras.backend.clear_session()
    gc.collect()
    for _device in ("GPU:0", "GPU:1"):
        try:
            tf.config.experimental.reset_memory_stats(_device)
        except Exception:
            pass
    print(f"[INIT] {modality}: BEFORE manifest load", flush=True)
    manifest = load_active_ceaset_manifest(modality=modality)
    print(f"[INIT] {modality}: AFTER manifest load rows={len(manifest)}", flush=True)
    print(f"[INIT] {modality}: BEFORE stats load", flush=True)
    stats = pd.read_csv(CEASET_STATS_CSV)
    print(f"[INIT] {modality}: AFTER stats load rows={len(stats)}", flush=True)

    training_manifest = manifest[
        manifest["split"].astype(str) == "training"
    ].reset_index(drop=True)
    eval_manifest = manifest[
        manifest["split"].astype(str) == "eval"
    ].reset_index(drop=True)

    if training_manifest.empty:
        raise RuntimeError("No training chunks found.")
    if eval_manifest.empty:
        raise RuntimeError("No evaluation chunks found.")

    sample_row = training_manifest.iloc[0]
    chunk_shape = tuple(
        load_chunk(
            sample_row[f"{modality}_{CFG.echo_labels[0]}_path"]
        ).shape
    )

    print(f"[INIT] {modality}: BEFORE trainer construction chunk_shape={chunk_shape} {_gpu_memory_string()}", flush=True)
    trainer = CEASETTrainer(modality, chunk_shape)
    print(f"[INIT] {modality}: AFTER trainer construction {_gpu_memory_string()}", flush=True)
    print(f"[INIT] {modality}: BEFORE maybe_resume()", flush=True)
    start_epoch, resume_mode = trainer.maybe_resume()
    print(f"[INIT] {modality}: AFTER maybe_resume() start_epoch={start_epoch} mode={resume_mode} {_gpu_memory_string()}", flush=True)

    outdir = TRAINING_ROOT / modality
    outdir.mkdir(parents=True, exist_ok=True)
    history_path = outdir / f"history_{RUN_NAME}.csv"

    # Preserve old epoch history instead of overwriting it on resume.
    history = []
    if history_path.is_file():
        try:
            old_history = pd.read_csv(history_path)
            if "epoch" in old_history.columns:
                old_history = old_history[
                    old_history["epoch"].astype(int) <= int(start_epoch)
                ].copy()
                if "validation_protocol" not in old_history.columns:
                    old_history["validation_protocol"] = "legacy_random_v1"
                history = old_history.to_dict("records")
        except Exception as exc:
            print(f"[WARN] Could not reuse old history CSV: {exc}")

    expected_train_batches = expected_direct_samples(
        manifest,
        "training",
        random_chunk_sample=TRAINING_RANDOM_CHUNK_SAMPLE,
    )
    expected_eval_batches = expected_direct_samples(
        manifest,
        "eval",
        random_chunk_sample=EVAL_MAX_CHUNKS,
    )

    print(
        "\n============================================================\n"
        f"[START] CEASET {modality.upper()}\n"
        f"Resume mode:             {resume_mode}\n"
        f"Starting after epoch:    {start_epoch}\n"
        f"Target final epoch:      {int(epochs)}\n"
        f"Training batch size:     {TRAINING_BATCH_SIZE}\n"
        f"Training batches/epoch:  {expected_train_batches}\n"
        f"Fixed eval batch size:   {TRAINING_EVAL_BATCH_SIZE}\n"
        f"Fixed eval batches:      {expected_eval_batches}\n"
        f"Training data format:    {'FAST .npy' if TRAINING_USE_FAST_NPY_CACHE else 'HDF5/NIfTI'}\n"
        f"Input pipeline:          DIRECT PYTHON (no tf.data)\n"
        f"Background workers:      0\n"
        f"Prefetch queue:          disabled\n"
        f"Dynamic echo axis:       {TRAINING_DYNAMIC_ECHO_AXIS}\n"
        f"Compiled train steps:    {TRAINING_COMPILE_STEPS}\n"
        f"Adversarial mode:        {TRAINING_GAN_MODE}\n"
        f"Generator device:        {GENERATOR_DEVICE}\n"
        f"Critic + GP device:      {CRITIC_DEVICE}\n"
        f"Gradient penalty lambda: {WGAN_GP_LAMBDA:g}\n"
        f"GP downsample factor:    {WGAN_GP_DOWNSAMPLE_FACTOR}\n"
        f"Critic updates/batch:    {CRITIC_UPDATES_BY_MODALITY[modality]}\n"
        f"Perceptual loss:         {TRAINING_ENABLE_PERCEPTUAL_LOSS}\n"
        f"Edge loss:               {TRAINING_ENABLE_EDGE_LOSS}\n"
        f"Coarse physics loss:     {TRAINING_ENABLE_COARSE_PHYSICS_LOSS}\n"
        f"Physics blur sigma:      {PHYSICS_GAUSSIAN_SIGMA_VOXELS}\n"
        f"Continuous aux TE:       {CONTINUOUS_AUXILIARY_TRAINING} {CONTINUOUS_TARGET_TRAIN_RANGE_MS} ms\n"
        "============================================================",
        flush=True,
    )

    preview_writer = PreviewWriter(os.environ.get('ECHO_GEN_PREVIEW_URL'), os.environ.get('ECHO_GEN_PREVIEW_TOKEN'), TRAINING_PREVIEW_INTERVAL_SECONDS)
    preview_axes = {}
    for epoch in range(int(start_epoch) + 1, int(epochs) + 1):
        check_training_stop()
        critic_updates = trainer.critic_updates_for_epoch(epoch)
        if not TRAINING_USE_DIRECT_PYTHON_LOOP:
            raise RuntimeError(
                "This stability revision intentionally requires "
                "TRAINING_USE_DIRECT_PYTHON_LOOP=True."
            )

        iterator = iter_ceaset_samples_direct(
            manifest,
            stats,
            modality,
            "training",
            epoch,
            random_chunk_sample=TRAINING_RANDOM_CHUNK_SAMPLE,
            shuffle=True,
            augment=True,
        )

        sums = defaultdict(float)
        n = 0
        epoch_start = time.time()
        last_log = epoch_start

        print(
            f"\n[{modality.upper()}] epoch={epoch:03d}/{int(epochs):03d} START "
            f"critic_updates={critic_updates} {_gpu_memory_string()}",
            flush=True,
        )

        while True:
            # Time data wait separately from GPU compute. The first three batches
            # are always printed so an I/O bottleneck cannot hide for hours.
            check_training_stop()
            data_wait_start = time.perf_counter()
            try:
                inputs, real, sample_meta = next(iterator)
            except StopIteration:
                break
            data_wait_s = time.perf_counter() - data_wait_start

            check_training_stop()
            preview_axis = -1
            if TRAINING_PREVIEW_ENABLED and preview_writer.due():
                case = str(sample_meta['case_id'])
                if case not in preview_axes:
                    try:
                        # Header only; chunk generation transposes source axes (1,0,2).
                        source = _prepared_case_path(case, modality, CFG.echo_labels[0], CFG)
                        preview_axes[case] = axial_chunk_axis(nib.load(str(source)).affine)
                    except Exception as exc:
                        preview_axes[case] = -1
                        print(f'[PREVIEW] Orientation unavailable for {case}: {exc}', flush=True)
                preview_axis = preview_axes[case]
            gpu_start = time.perf_counter()
            batch_metrics = trainer.train_batch(
                inputs,
                real,
                critic_updates,
                preview_axis=preview_axis,
            )
            preview_tensor = batch_metrics.pop('_preview', None)
            # One synchronization point after the complete GPU batch.
            batch_metrics_np = {
                key: float(value.numpy())
                for key, value in batch_metrics.items()
            }
            gpu_step_s = time.perf_counter() - gpu_start
            if preview_tensor is not None:
                try:
                    preview_writer.submit(preview_tensor.numpy(), {
                        'run_name': RUN_NAME, 'modality': modality, 'epoch': epoch, 'batch': n+1,
                        'case_id': str(sample_meta['case_id']), 'target_te_ms': float(sample_meta['target_te_ms']),
                        'target_echo': str(sample_meta['target_echo']),
                        'input_tes_ms': [float(te) for te in sample_meta['input_tes_ms']],
                        'input_echoes': [str(echo) for echo in sample_meta['input_echoes']],
                        'regime': str(sample_meta['regime']),
                        'plane': 'transversal', 'chunk_axis': preview_axis,
                        'slice_index': int(real.shape[preview_axis+1])//2,
                        'source': 'first critic update / existing generator output',
                    })
                except Exception as exc:
                    print(f'[PREVIEW] Skipped: {exc}', flush=True)
                del preview_tensor

            for key, value in batch_metrics_np.items():
                sums[key] += value
            n += 1

            if n <= 3:
                print(
                    f"[{modality.upper()}] FIRST-BATCH-DIAG "
                    f"B{n}: case={sample_meta.get('case_id','?')} "
                    f"K={int(inputs['echo_images'].shape[1])} "
                    f"data_wait={data_wait_s:.3f}s "
                    f"gpu_train_step={gpu_step_s:.3f}s "
                    f"{_gpu_memory_string()}",
                    flush=True,
                )

            now = time.time()
            batch_trigger = (
                int(log_every_batches) > 0
                and n % int(log_every_batches) == 0
            )
            time_trigger = (
                float(log_every_minutes) > 0
                and (now - last_log) >= float(log_every_minutes) * 60.0
            )

            if batch_trigger or time_trigger:
                elapsed = now - epoch_start
                sec_per_batch = elapsed / max(n, 1)
                remaining = max(expected_train_batches - n, 0)
                eta = remaining * sec_per_batch
                pct = 100.0 * n / max(expected_train_batches, 1)

                print(
                    f"[{modality.upper()}] "
                    f"E{epoch:03d} B{n:04d}/{expected_train_batches:04d} "
                    f"{pct:5.1f}% | elapsed={elapsed/60:6.1f}min "
                    f"ETA≈{eta/60:6.1f}min | "
                    f"G={sums['g_total']/n:.5f} "
                    f"rec={sums['g_rec']/n:.5f} "
                    f"recW={sums['g_rec_weighted']/n:.4f} "
                    f"adv={sums['g_adv']/n:.3e} "
                    f"advW={sums['g_adv_weighted']/n:.3e} "
                    f"phys={sums['g_phys']/n:.3e} "
                    f"physI={sums['g_phys_intensity']/n:.3e} "
                    f"physS={sums['g_phys_structure']/n:.3e} "
                    f"physW={sums['g_phys_weighted']/n:.3e} "
                    f"auxW={sums['g_aux_phys_weighted']/n:.3e} "
                    f"D={sums['d_loss']/n:.3e} "
                    f"Wgap={sums['d_gap']/n:.3e} "
                    f"GP={sums['gp']/n:.3e} "
                    f"| {sec_per_batch:.1f}s/batch | {_gpu_memory_string()}",
                    flush=True,
                )
                last_log = now

        train_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "seconds": train_seconds,
            "batches": n,
            "critic_updates": critic_updates,
            "resume_mode": resume_mode,
            "validation_protocol": "fixed_v2",
            **{key: value / max(n, 1) for key, value in sums.items()},
        }

        # -------------------------------------------------------------
        # FIXED validation: same chunk order + same echo configurations
        # on every epoch, so eval_MAE is now directly comparable.
        # -------------------------------------------------------------
        print(
            f"[{modality.upper()}] epoch={epoch:03d} fixed validation START",
            flush=True,
        )
        eval_start = time.time()
        eval_result = evaluate_ceaset_direct(
            trainer.generator,
            manifest,
            stats,
            modality,
            max_chunks=EVAL_MAX_CHUNKS,
            seed=FIXED_EVAL_SEED,
            shuffle=FIXED_EVAL_SHUFFLE,
            log_every_batches=25,
            label=f"{modality.upper()} FIXED-EVAL E{epoch:03d}",
        )
        eval_seconds = time.time() - eval_start
        row.update({f"eval_{key}": value for key, value in eval_result.items()})
        row["eval_seconds"] = eval_seconds

        history = [h for h in history if int(h.get("epoch", -1)) != int(epoch)]
        history.append(row)
        history_df = pd.DataFrame(history).sort_values("epoch").reset_index(drop=True)
        history_tmp = history_path.with_suffix(".tmp")
        history_df.to_csv(history_tmp, index=False)
        history_tmp.replace(history_path)

        # Fixed validation makes best-checkpoint tracking meaningful.
        fixed_mask = (
            history_df["validation_protocol"].astype(str) == "fixed_v2"
            if "validation_protocol" in history_df.columns
            else np.zeros(len(history_df), dtype=bool)
        )
        finite_history = history_df[
            fixed_mask & np.isfinite(history_df["eval_mae"].astype(float))
        ]
        best_epoch = None
        best_mae = np.nan
        if not finite_history.empty:
            best_idx = finite_history["eval_mae"].astype(float).idxmin()
            best_epoch = int(finite_history.loc[best_idx, "epoch"])
            best_mae = float(finite_history.loc[best_idx, "eval_mae"])

        print(
            f"[{modality.upper()}] epoch={epoch:03d} COMPLETE | "
            f"G={row['g_total']:.5f} rec={row['g_rec']:.5f} "
            f"adv={row['g_adv']:.3e} advW={row['g_adv_weighted']:.3e} "
            f"phys={row['g_phys']:.3e} "
            f"(I={row['g_phys_intensity']:.3e},S={row['g_phys_structure']:.3e}) "
            f"physW={row['g_phys_weighted']:.3e} "
            f"auxW={row['g_aux_phys_weighted']:.3e} "
            f"D={row['d_loss']:.3e} Wgap={row['d_gap']:.3e} "
            f"GP={row['gp']:.3e} "
            f"fixed_eval_MAE={row['eval_mae']:.5f} "
            f"train={train_seconds/60:.1f}min eval={eval_seconds/60:.1f}min "
            f"| best={best_epoch}:{best_mae:.5f}",
            flush=True,
        )

        if epoch % int(TRAINING_SAVE_EVERY_EPOCHS) == 0:
            gp_path, dp_path, state_path = trainer.save_checkpoint(epoch)
            print(f"[SAVE] generator: {gp_path}", flush=True)
            print(f"[SAVE] critic:    {dp_path}", flush=True)
            print(f"[SAVE] full state:{state_path}", flush=True)

        # Save a convenience "best" generator whenever the CURRENT epoch is best.
        if best_epoch == int(epoch):
            best_path = TRAINING_MODEL_ROOT / modality / f"generator_{RUN_NAME}_best_fixed_eval.weights.h5"
            trainer.generator.save_weights(best_path)
            (TRAINING_MODEL_ROOT / modality / f"best_fixed_eval_{RUN_NAME}.txt").write_text(
                f"epoch={epoch}\neval_mae={row['eval_mae']:.10f}\n",
                encoding="utf-8",
            )

        # -------------------------------------------------------------
        # Early stopping on the FIXED validation set.
        # A meaningful improvement must exceed EARLY_STOPPING_MIN_DELTA.
        # -------------------------------------------------------------
        if ENABLE_EARLY_STOPPING and epoch >= int(EARLY_STOPPING_START_EPOCH):
            fixed_mask_es = (
                history_df["validation_protocol"].astype(str) == "fixed_v2"
                if "validation_protocol" in history_df.columns
                else np.zeros(len(history_df), dtype=bool)
            )
            eval_hist = history_df[
                fixed_mask_es & np.isfinite(history_df["eval_mae"].astype(float))
            ][["epoch", "eval_mae"]].copy()
            eval_hist["epoch"] = eval_hist["epoch"].astype(int)
            eval_hist["eval_mae"] = eval_hist["eval_mae"].astype(float)

            # Walk chronologically and track the last epoch that improved the
            # incumbent best by at least min_delta.
            incumbent = np.inf
            last_meaningful_improvement = None
            for _, hist_row in eval_hist.sort_values("epoch").iterrows():
                value = float(hist_row["eval_mae"])
                if value < incumbent - float(EARLY_STOPPING_MIN_DELTA):
                    incumbent = value
                    last_meaningful_improvement = int(hist_row["epoch"])

            if last_meaningful_improvement is not None:
                stagnant_epochs = int(epoch) - int(last_meaningful_improvement)
                if stagnant_epochs >= int(EARLY_STOPPING_PATIENCE):
                    print(
                        f"[EARLY STOP] {modality.upper()} epoch={epoch}: "
                        f"no fixed-eval improvement >= {EARLY_STOPPING_MIN_DELTA:g} "
                        f"for {stagnant_epochs} epochs. "
                        f"Best meaningful epoch={last_meaningful_improvement}, "
                        f"best fixed eval≈{incumbent:.5f}.",
                        flush=True,
                    )
                    gc.collect()
                    break

        gc.collect()

    check_training_stop()
    # Always retain the final completed epoch, including early stopping between save intervals.
    if history:
        final_epoch = max(int(item['epoch']) for item in history)
        if final_epoch > int(start_epoch) and final_epoch % int(TRAINING_SAVE_EVERY_EPOCHS) != 0:
            trainer.save_checkpoint(final_epoch)
    result_df = pd.DataFrame(history).sort_values("epoch").reset_index(drop=True)

    global CEASET_PRODUCTION_TRAINERS_COMPLETED
    CEASET_PRODUCTION_TRAINERS_COMPLETED += 1
    print(
        f"[CLEANUP] {modality}: production trainer completed; "
        f"completed={CEASET_PRODUCTION_TRAINERS_COMPLETED}. "
        "Releasing model/optimizer objects before the next modality.",
        flush=True,
    )
    preview_writer.close()
    del trainer
    tf.keras.backend.clear_session()
    gc.collect()
    return result_df
