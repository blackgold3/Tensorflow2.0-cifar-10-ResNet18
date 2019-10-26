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

# 训练 #
model = resnet.ResNet([2, 2, 2, 2], 10)
optimizer = tf.keras.optimizers.Nadam(0.001)
iter = iter(valid_db)
for i in range(10):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as type:
            out = model(x)
            loss = tf.keras.losses.categorical_crossentropy(y, out, from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = type.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    (x_va, y_va) = next(iter)
    pred = model(x_va)
    loss = tf.keras.losses.categorical_crossentropy(y_va, pred, from_logits=True)
    loss = tf.reduce_mean(loss)
    print("第%d次迭代，验证集的损失为%f"%(i, loss))
# 训练 #

# 测试 #
model.evaluate(test_db)
# 测试 #