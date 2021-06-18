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

def get_resnet_model(class_, filters, block_size, num_tilt_classes, num_pan_classes, reg_lambda=0.0, fdropout=False):
    input = tf.keras.Input(
        shape=(64,64,3)
    )

    assert class_ == "tilt" or class_ == "pan"

    x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                   padding="same")(input)
  
    x = tf.keras.layers.BatchNormalization()(x)
  
    for nFilters, nBlocks in zip(filters, block_size):
       x = ResidualBlock(nFilters, stride=2, reg_lambda=reg_lambda)(x)

    for _ in range(1, nBlocks):
        x = ResidualBlock(nFilters, stride=1, reg_lambda=reg_lambda)(x)

    # Final part
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    if fdropout:
       x = tf.keras.layers.Dropout(0.5)(x)
  
    if class_ == "tilt":
        output = tf.keras.layers.Dense(num_tilt_classes, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda), kernel_initializer="he_normal")(x)
    else:
        output = tf.keras.layers.Dense(num_pan_classes, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda), kernel_initializer="he_normal")(x)

    return tf.keras.Model(input, output)

def get_bottleneck_resnet_model(class_, filters, block_size, num_tilt_classes, num_pan_classes, reg_lambda=0.0, fdropout=False):
    input = tf.keras.Input(
        shape=(64,64,3)
    )

    assert class_ == "tilt" or class_ == "pan"

    # Additional batch normalisation layer
    x = tf.keras.layers.BatchNormalization()(input)

    x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(5, 5),
                                   strides=1,
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                   padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.ZeroPadding2D((1,1))(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)

    x = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=tf.keras.regularizers.l2(reg_lambda),
                                   padding="same")(input)
  

  
    for nFilters, nBlocks in zip(filters, block_size):
        # Reduce dimensions in these blocks
        stride = 2
        if nFilters == filters[0]:
            stride = 1
        x = BottleneckResidualBlock(nFilters, stride=stride, reg_lambda=reg_lambda, dropout=fdropout)(x)

        for _ in range(1, nBlocks):
            x = BottleneckResidualBlock(nFilters, stride=1, reg_lambda=reg_lambda, dropout=fdropout)(x)
  
    # Final part
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
  
    if class_ == "tilt":
        output = tf.keras.layers.Dense(num_tilt_classes, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda), kernel_initializer="he_normal")(x)
    else:
        output = tf.keras.layers.Dense(num_pan_classes, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(reg_lambda), kernel_initializer="he_normal")(x)

    return tf.keras.Model(input, output)

def get_callbacks(name, logdir, early_stop=True):
    print(name)
    if early_stop:
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_CategoricalCrossentropy', patience=25),
            tf.keras.callbacks.TensorBoard(logdir/name, histogram_freq=60, embeddings_freq=60)
        ]
    else:
        return [tf.keras.callbacks.TensorBoard(logdir/name)]
