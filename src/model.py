import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

@tf.keras.utils.register_keras_serializable()
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha)
        context = x * alpha
        return tf.keras.backend.sum(context, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        return config


def build_model(hp,
                vocab_size: int = 128,
                max_seq_len: int = 75,
                embedding_dim: int = 64,
                cnn_filters: int = 128,
                cnn_kernel_size: int = 3,
                lstm_units: int = 64,
                dense_units: int = 64,
                use_bidirectional: bool = True,
                default_dropout: float = 0.3,
                default_lr: float = 0.001,
                lr_min: float = 0.0001,
                lr_max: float = 0.01,
                dropout_min: float = 0.2,
                dropout_max: float = 0.5):
    """Build and compile the 1D-CNN + Bi-LSTM model.

    When *hp* is a KerasTuner ``HyperParameters`` object the function
    will define tuneable search spaces.  When *hp* is ``None`` the
    default values are used (useful for standalone evaluation).

    Parameters
    ----------
    hp : keras_tuner.HyperParameters or None
        Hyperparameter handle from KerasTuner.  Pass ``None`` to use
        the default values supplied in the remaining arguments.
    vocab_size : int
        Number of unique tokens (characters) + padding/OOV.
    max_seq_len : int
        Fixed input sequence length.
    embedding_dim : int
        Dimensionality of the character-level embedding.
    cnn_filters : int
        Number of filters in the Conv1D layer.
    cnn_kernel_size : int
        Kernel size for the Conv1D layer.
    lstm_units : int
        Number of units in the LSTM cell.
    dense_units : int
        Number of neurons in the fully-connected hidden layer.
    use_bidirectional : bool
        If True wrap the LSTM in a ``Bidirectional`` wrapper.
    default_dropout, default_lr : float
        Fallback values when *hp* is None.
    lr_min, lr_max : float
        KerasTuner learning-rate search range.
    dropout_min, dropout_max : float
        KerasTuner dropout search range.

    Returns
    -------
    tensorflow.keras.Model
        Compiled Keras model ready for training.
    """

    # ---- Tuneable hyper-parameters ------------------------------------------
    if hp is not None:
        dropout_rate = hp.Float("dropout_rate",
                                min_value=dropout_min,
                                max_value=dropout_max,
                                step=0.05)
        learning_rate = hp.Float("learning_rate",
                                 min_value=lr_min,
                                 max_value=lr_max,
                                 sampling="log")
    else:
        dropout_rate = default_dropout
        learning_rate = default_lr

    # ---- Model definition ---------------------------------------------------
    inputs = layers.Input(shape=(max_seq_len,), name="char_input")

    # Layer 1 – Character-level Embedding (trainable)
    x = layers.Embedding(input_dim=vocab_size,
                         output_dim=embedding_dim,
                         input_length=max_seq_len,
                         name="char_embedding")(inputs)

    # Layer 2 – 1D Convolutional Layer
    x = layers.Conv1D(filters=cnn_filters,
                      kernel_size=cnn_kernel_size,
                      activation="relu",
                      padding="same",
                      name="conv1d")(x)

    # Layer 3 – Max Pooling
    x = layers.MaxPooling1D(pool_size=2, name="maxpool1d")(x)

    # Layer 4 – LSTM / Bi-LSTM
    # V3 Upgrade: return_sequences=True to pass full sequence to Attention layer
    lstm_layer = layers.LSTM(lstm_units, return_sequences=True, name="lstm")
    if use_bidirectional:
        x = layers.Bidirectional(lstm_layer, name="bi_lstm")(x)
    else:
        x = lstm_layer(x)

    # Layer 4.5 – V3 Custom Attention Layer
    # Solves Flaw 1: Allows model to focus on the suffix instead of overreacting to subdomain entropy
    x = Attention(name="attention")(x)

    # Layer 5 – Fully-connected Dense layer
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)

    # Layer 6 – Dropout for regularisation
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # Layer 7 – Output layer
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DGA_Detector_V3")

    # ---- Compilation --------------------------------------------------------
    # V-Final Upgrade: Use BinaryFocalCrossentropy to force the AI to penalize 
    # hard misclassifications (like our injected edge cases) more than easy ones.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )

    logger.info("Model compiled (lr=%.6f, dropout=%.2f). Custom Attention & Focal Loss included.",
                learning_rate, dropout_rate)
    return model
