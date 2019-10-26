import tensorflow as tf
import resnet

# tf2 处理无法读取卷积的问题 #
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# tf2 处理无法读取卷积的问题 #

# 预处理 #
def pre_process(x, y):
    x = tf.multiply(tf.cast(x, tf.float32), 2) / 255.0 - 1.0
    y = tf.cast(y, tf.int32)
    return x, y
# 预处理 #

# 导入数据  #
(x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y = tf.one_hot(y, depth=10)
y = tf.squeeze(y, axis=1)
y_test = tf.one_hot(y_test, depth=10)
y_test = tf.squeeze(y_test, axis=1)
x_valid = x[40000::, ...]   # 分出一个验证集
y_valid = y[40000::, ...]
x = x[0:40000:1, ...]       # 切割掉验证集
y = y[0:40000:1, ...]
# 导入数据  #

# 建立迭代器 #
batch_size = 100
train_db = tf.data.Dataset.from_tensor_slices((x,y))
valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = train_db.map(pre_process).shuffle(40000).batch(batch_size)
valid_db = valid_db.map(pre_process).batch(batch_size)
test_db = test_db.map(pre_process).batch(batch_size)
# 建立迭代器 #

# 加入模型 #
t = iter(train_db)
net = resnet.ResNet([2, 2, 2, 2], 10)
net.compile(tf.keras.optimizers.Nadam(0.001),
            loss=tf.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
net.fit(train_db, epochs=10, validation_data=valid_db, validation_freq=1)
net.evaluate(test_db)
net.save_weights('ckpt/weights.ckpt')
# net.build((None, 32, 32, 3))
# net.load_weights('ckpt/weights.ckpt')
# net.summary()
# net.evaluate(test_db)
# 加入模型 #
