import tensorflow as tf

assert tf.__version__[0] == '2'
AUTOTUNE = tf.data.experimental.AUTOTUNE

class ResidualBlock(tf.keras.layers.Layer):
    # Initialize components of the model
    def __init__(self, filter_num, stride=1, reg_lambda=0.0, dropout=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       kernel_initializer="he_normal",
                                                       kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    # Define the forward function
    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):

        config = super().get_config().copy()
        config.update({
          'conv1': self.conv1,
          'bn1': self.bn1,
          'conv2': self.conv2,
          'bn2': self.bn2,
          'downsample': self.downsample,
        })
        return config

class BottleneckResidualBlock(tf.keras.layers.Layer):
    # Includes preactivation

    # Initialize components of the model
    def __init__(self, filter_num, stride=1, reg_lambda=0.0, dropout=False):
        super(BottleneckResidualBlock, self).__init__()

        self.dropout = dropout
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                    kernel_size=(1, 1),
                                    use_bias=False,
                                    strides=1,
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                    padding="same")
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            use_bias=False,
                                            strides=stride, # Use stride in the 3x3 instead of 1x1 convolution layer
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                            padding="same")

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num,
                                    kernel_size=(1, 1),
                                    use_bias=False,
                                    strides=1,
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                    padding="same")

        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       kernel_initializer="he_normal",
                                                       kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    # Define the forward function
    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        # Use preactivation structure for each convolution (batch_norm, activation, conv)
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x) # Stride at this layer instead of at 1x1
        if self.dropout:
            x = self.dropout1(x)

        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):

        config = super().get_config().copy()
        config.update({
          'bn1': self.bn1,
          'conv1': self.conv1,
          'bn2': self.bn2,
          'conv2': self.conv2,
          'dropout': self.dropout1,
          'bn3': self.bn3,
          'conv3': self.conv3,
          'downsample': self.downsample,
        })
        return config

