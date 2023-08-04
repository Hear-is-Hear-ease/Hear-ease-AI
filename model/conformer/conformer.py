import tensorflow as tf
from einops import rearrange
from einops.layers.tensorflow import Rearrange

from .attention import Attention


class Swish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)


class GLU(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.dim = dim

    def call(self, inputs):
        out, gate = tf.split(inputs, 2, axis=self.dim)
        return out * tf.sigmoid(gate)


class DepthwiseLayer(tf.keras.layers.Layer):
    def __init__(self, chan_in, chan_out, kernel_size, padding, **kwargs):
        super(DepthwiseLayer, self).__init__(**kwargs)
        self.padding = padding
        self.chan_in = chan_in
        self.conv = tf.keras.layers.Conv1D(chan_out, 1, groups=chan_in)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1])
        padded = tf.zeros(
            [self.chan_in * self.chan_in] - tf.shape(inputs), dtype=inputs.dtype
        )
        inputs = tf.concat([inputs, padded], 0)
        inputs = tf.reshape(inputs, [-1, self.chan_in, self.chan_in])

        return self.conv(inputs)


class Scale(tf.keras.layers.Layer):
    def __init__(self, scale, fn, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.scale = scale
        self.fn = fn

    def call(self, inputs, **kwargs):
        return self.fn(inputs, **kwargs) * self.scale


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn, **kwargs):
        super(PreNorm, self).__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.fn = fn

    def call(self, inputs, **kwargs):
        inputs = self.norm(inputs)
        return self.fn(inputs, **kwargs)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4, dropout=0.0, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim * mult, activation=Swish()),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dim, input_dim=dim * mult),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, inputs):
        return self.net(inputs)


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, causal, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.causal = causal

    def call(self, inputs):
        if not self.causal:
            return tf.keras.layers.BatchNormalization(axis=-1)(inputs)
        return tf.identity(inputs)


class ConformerConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.0,
        **kwargs
    ):
        super(ConformerConvModule, self).__init__(**kwargs)

        inner_dim = dim * expansion_factor
        if not causal:
            padding = (kernel_size // 2, kernel_size //
                       2 - (kernel_size + 1) % 2)
        else:
            padding = (kernel_size - 1, 0)

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(axis=-1),
                Rearrange("b n c -> b c n"),
                tf.keras.layers.Conv1D(filters=inner_dim * 2, kernel_size=1),
                GLU(dim=1),
                DepthwiseLayer(
                    inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
                ),
                BatchNorm(causal=causal),
                Swish(),
                tf.keras.layers.Conv1D(filters=dim, kernel_size=1),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, inputs):
        return self.net(inputs)


class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        **kwargs
    ):
        super(ConformerBlock, self).__init__(**kwargs)
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs, mask=None):
        inputs = self.ff1(inputs) + inputs
        inputs = self.attn(inputs, mask=mask) + inputs
        inputs = self.conv(inputs) + inputs
        inputs = self.ff2(inputs) + inputs
        inputs = self.post_norm(inputs)
        return inputs


# 기존 클래스들과 함수들은 동일하게 유지합니다.
class ConformerModel(tf.keras.Model):
    def __init__(self, num_blocks, dim, dim_head=64, heads=8, ff_mult=4,
                 conv_expansion_factor=2, conv_kernel_size=31, attn_dropout=0.0,
                 ff_dropout=0.0, conv_dropout=0.0, num_classes=10):
        super(ConformerModel, self).__init__()

        self.blocks = [ConformerBlock(dim=dim, dim_head=dim_head, heads=heads,
                                      ff_mult=ff_mult, conv_expansion_factor=conv_expansion_factor,
                                      conv_kernel_size=conv_kernel_size, attn_dropout=attn_dropout,
                                      ff_dropout=ff_dropout, conv_dropout=conv_dropout)
                       for _ in range(num_blocks)]

        self.post_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(
            num_classes, activation='softmax')

    def call(self, inputs, mask=None):
        for block in self.blocks:
            inputs = block(inputs, mask=mask)

        inputs = self.post_norm(inputs)
        inputs = self.pooling(inputs)
        outputs = self.classifier(inputs)

        return outputs


if __name__ == '__main__':
    num_blocks = 6
    num_classes = 10

    # 임의의 MFCC 데이터 생성 (실제 데이터를 사용해야 함)
    mfcc_data = np.array([tf.random.normal([1, 1024, 512])
                          for _ in range(10)])  # 1024 개의 프레임에 대한 512차원 MFCC
    print(mfcc_data.shape)

    # Conformer 모델 생성
    model = ConformerModel(num_blocks=num_blocks,
                           dim=512,
                           conv_expansion_factor=2,
                           conv_kernel_size=31,
                           attn_dropout=0.1,
                           ff_dropout=0.1,
                           conv_dropout=0.1,
                           num_classes=num_classes)

    # 입력 데이터를 1차원으로 변환 (Conformer 모델은 1차원 시퀀스를 처리)
    mfcc_data_flattened = tf.reshape(mfcc_data, [1, -1, 512])

    # 모델에 입력 데이터 통과
    y = model(mfcc_data_flattened)

    print(y.shape)  # 출력 예시: (1, num_classes)
