import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = r"--xla_gpu_cuda_data_dir='D:\ProgramData\miniconda3\envs\your_conda_env'"
os.environ["MLIR_CRASH_REPRODUCER_DIRECTORY"] = r"E:\specifying_a_folder"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
import tensorflow as tf
from tensorflow.keras import backend as K
import spektral
import math

# ******** Custom callbacks ********
class Weight(tf.keras.callbacks.Callback):    # DWA
    def __init__(self, w0, w1):
        self.w0, self.w1 = w0, w1
        self.K = 3  # 3 losses (CE, MKMMD, TVGNN)
        self.Loss = [list() for i in range(self.K)]
        self.L, self.S, self.W = float('-inf'), [], []
        self.DWA = False
        super(Weight, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.Loss[0].append(logs.get("label_loss"))
        self.Loss[1].append(logs.get("MMD_loss"))
        try:
            self.Loss[2].append(tf.math.reduce_sum(self.model.layers[self.L].losses) /
                                self.model.layers[self.L].submodules[self.S[0]].get_weights()[self.W[0]])   # run_eagerly=True
        except:
            self.Loss[2].append((logs["loss"] - self.w0*logs["label_loss"] - self.w1*logs["MMD_loss"]) /
                                self.model.layers[self.L].submodules[self.S[0]].get_weights()[self.W[0]])   # run_eagerly=False
        self.DWA = True if logs.get("label_sparse_categorical_accuracy") > 0.7 and epoch >= 1 else False

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            for l in range(len(self.model.layers)):
                if self.model.layers[l].name == 'Total_Variation_Graph_Neural_Network':
                    self.L = l # One total variation graph neural network

            for s in range(len(self.model.layers[self.L].submodules)):
                if self.model.layers[self.L].submodules[s].name == 'Asymmetric_Cheeger_Cut_Pooling':
                    self.S.append(s)    # Three asymmetric cheeger cut pooling layers

            for w in range(len(self.model.layers[self.L].submodules[self.S[0]].weights)):
                if self.model.layers[self.L].submodules[self.S[0]].weights[w].name.find('totvar_coeff') +1 \
                    or self.model.layers[self.L].submodules[self.S[0]].weights[w].name.find('balance_coeff') +1:
                        self.W.append(w)    # Two coefficients (totvar & balance)

        T = 1.
        if self.DWA:
            wKt = list()
            for k in range(self.K):
                Lkt1, Lkt2 = self.Loss[k][epoch-1], self.Loss[k][epoch-2]
                wKt.append(Lkt1 / Lkt2)

            lambdaKt = list()
            wKt = [wkt/T - max(wKt)/T for wkt in wKt]
            for wkt in wKt:
                lambdaKt.append(self.K * math.exp(wkt) / sum(math.exp(wkt) for wkt in wKt))

            K.set_value(self.w0, lambdaKt[0]);  K.set_value(self.w1, lambdaKt[1])
            for s in self.S:
                weights = []
                for w in range(len(self.model.layers[self.L].submodules[s].weights)):
                    try:
                        self.W.index(w)
                        weights.append(lambdaKt[2])
                    except ValueError:
                        weights.append(self.model.layers[self.L].submodules[s].get_weights()[w])
                self.model.layers[self.L].submodules[s].set_weights(weights)

        print(f'Loss weights (CE, MKMMD, TVGNN): \
              {K.get_value(self.w0):.4f} {K.get_value(self.w1):.4f} {self.model.layers[self.L].submodules[self.S[0]].get_weights()[self.W[0]]:.4f}')

def learning_rate(Epochs, lr_init):
    Epochs = Epochs -1
    def scheduler(epoch, lr):
        alpha, beta = 15., 0.5
        return lr_init / (1. + alpha* epoch/Epochs) **beta
    return tf.keras.callbacks.LearningRateScheduler(scheduler)

# ******** Custom layers ********
def squareplus(x):
    b = 4
    return 0.5* (tf.math.sqrt(tf.math.pow(x,2) +b) +x)

class generator(tf.keras.layers.Layer):
    def __init__(self, features):
        super(generator, self).__init__(trainable=True)
        self.features = features

    def build(self, input_shape):
        self.layers = 3
        self.dense = [tf.keras.layers.Dense(self.features, activation=squareplus, kernel_initializer='he_uniform')
                      for _ in range(self.layers)]

    def call(self, noise):
        for l in range(self.layers):
            noise = self.dense[l](noise)
        return noise

class TVGNN(tf.keras.layers.Layer):
    def __init__(self, features, domains):
        self.features, self.domains = features, domains
        super(TVGNN, self).__init__(trainable=True, name="Total_Variation_Graph_Neural_Network")

    def build(self, input_shape):
        nodes = input_shape[0][1]

        self.stack = 1
        self.combs = 4
        if round(nodes / 2**(self.combs-1)) < 2:
            self.combs = math.log2(nodes/2) +1
        self.MP, self.Pool = [0 for i in range(self.combs)], [0 for i in range(self.combs)]

        for i in range(self.combs):
            self.MP[i] = [spektral.layers.GTVConv(self.features, delta_coeff=1.6, epsilon=1e-3, activation=squareplus, kernel_initializer='he_uniform')
                          for _ in range(self.stack)]

            if i != self.combs-1:
                self.Pool[i] = ACCPool(self.domains, self.combs-1, round(nodes / 2**(i+1)), mlp_hidden=2, totvar_coeff=1., balance_coeff=1.)
            else:
                self.Pool[i] = spektral.layers.GlobalAvgPool()

        super(TVGNN, self).build(input_shape)

    def call(self, X_A):
        X, A = X_A[0], X_A[1]

        for i in range(self.combs):
            for GTVConv in self.MP[i]:
                X = GTVConv([X, A])

            if i != self.combs-1:
                X, A = self.Pool[i]([X, A])
            else:
                X = self.Pool[i](X)

        return X

class ACCPool(spektral.layers.AsymCheegerCutPool):
    def __init__(self, domains, combsM1,
                 k,
                 mlp_hidden,
                 totvar_coeff=1.0, balance_coeff=1.0):
        super().__init__(k=k, return_selection=False,
                         trainable=True, name = "Asymmetric_Cheeger_Cut_Pooling")
        self.domains, self.combsM1 = domains, combsM1
        self.k = k
        self.mlp_hidden = mlp_hidden
        self.totvar_coeff, self.balance_coeff = totvar_coeff, balance_coeff

    def build(self, input_shape):
        self.Totvar_coeff = self.add_weight(name='totvar_coeff', shape=(), trainable=False,
                                            initializer=tf.constant_initializer(self.totvar_coeff))
        self.Balance_coeff = self.add_weight(name='balance_coeff', shape=(), trainable=False,
                                             initializer=tf.constant_initializer(self.balance_coeff))

        self.mlp = tf.keras.Sequential()
        features = input_shape[0][2]
        units = [int(features + (self.k-features) / (self.mlp_hidden+1) * (l+1)) for l in range(self.mlp_hidden)]
        for l in range(self.mlp_hidden):
            self.mlp.add(tf.keras.layers.Dense(units[l], activation=squareplus, kernel_initializer="he_uniform"))
        self.mlp.add(tf.keras.layers.Dense(self.k, activation="softmax", kernel_initializer="he_uniform"))

    def select(self, x, a, i, mask=None):
        s = self.mlp(x)
        if mask is not None:
            s *= mask[0]

        # Total variation & Asymmetric l1-norm losses
        tv_loss = self.totvar_loss(a, s)
        if K.ndim(a) == 3:
            tv_loss = K.mean(tv_loss)
        self.add_loss(0.5 * 1/self.domains/self.combsM1 * self.Totvar_coeff * tv_loss)

        bal_loss = self.balance_loss(s)
        if K.ndim(a) == 3:
            bal_loss = K.mean(bal_loss)
        self.add_loss(0.5 * 1/self.domains/self.combsM1 * self.Balance_coeff * bal_loss)

        return s

    def get_config(self):
        config = {"k": self.k, "mlp_hidden": self.mlp_hidden, "totvar_coeff": self.totvar_coeff, "balance_coeff": self.balance_coeff}
        base_config = super().get_config()
        return {**base_config, **config}

# ******** Custom loss ********
def compute_MK_MMD(source, target, kernel_mul=2., kernel_num=5):
    """
    Args:
        source (tf.float32): (num_source_sample, feature_dimension)
        target (tf.float32): (num_target_sample, feature_dimension)
    Returns:
        MK-MMD (tf.float32)
    """
    ns, nt = tf.shape(source)[0], tf.shape(target)[0]
    n = tf.cast(ns+nt, 'float32')

    total = tf.concat([source, target], axis=0)
    total0, total1 = tf.expand_dims(total, axis=0), tf.expand_dims(total, axis=1)
    L2_distance = tf.reduce_sum((total0 - total1) **2, axis=2)
    bandwidth = tf.reduce_sum(L2_distance) / (n**2 - n)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * kernel_mul**i for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    kernels = sum(kernel_val)

    XX = tf.reduce_sum(kernels[:ns, :ns]) / tf.cast(ns**2, 'float32')
    YY = tf.reduce_sum(kernels[-nt:, -nt:]) / tf.cast(nt**2, 'float32')
    XY = tf.reduce_sum(kernels[:ns, -nt:]) / tf.cast(ns*nt, 'float32')
    YX = tf.reduce_sum(kernels[-nt:, :ns]) / tf.cast(ns*nt, 'float32')
    return XX + YY - XY - YX

class MK_MMD(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        C = tf.shape(y_pred)[1] - tf.shape(y_true)[1]
        domains = tf.shape(y_pred)[1] -C
        MKMMD = tf.TensorArray(dtype='float32', size=domains)

        for i in range(domains):
            source = y_pred[:,i,:]
            l = tf.cast(y_true[:,i], 'int32')
            mkmmd = tf.TensorArray(dtype='float32', size=C, clear_after_read=False)
            for c in range(C):
                source_c = tf.gather(source, indices=tf.squeeze(tf.where(l==c)))
                if tf.size(tf.where(l==c)) < 2:
                    raise ValueError('The i-th source subject does not have enough samples belonging to the c-th attention class!')

                target_c = y_pred[:,domains+c,:]

                indices = tf.cast(tf.linspace(0, C-1, C), 'int32')
                condition = tf.not_equal(indices, c)
                indices = tf.boolean_mask(indices, condition)
                non_target_c = tf.gather(y_pred, axis=1, indices=domains+indices)

                mkmmd_target = compute_MK_MMD(source_c, target_c)
                mkmmd_target = mkmmd_target if mkmmd_target>0. else 1e-7

                mkmmd_non_target = tf.TensorArray(dtype='float32', size=C-1, clear_after_read=False)
                for c_1 in tf.range(C-1):
                    mkmmd_non_target_c = compute_MK_MMD(source_c, non_target_c[:,c_1,:])
                    mkmmd_non_target_c = mkmmd_non_target_c if mkmmd_non_target_c>0. else 1e-7
                    mkmmd_non_target = mkmmd_non_target.write(c_1, 1./mkmmd_non_target_c)

                mkmmd_non_target = mkmmd_non_target.stack()
                mkmmd_non_target = tf.math.reduce_sum(mkmmd_non_target)
                mkmmd = mkmmd.write(c, mkmmd_target+mkmmd_non_target)

            mkmmd = mkmmd.stack()
            mkmmd = tf.math.reduce_mean(mkmmd)
            MKMMD = MKMMD.write(i, mkmmd)

        MKMMD = MKMMD.stack()

        return tf.math.reduce_mean(MKMMD)

# ******** Creating a model ********
def AAD(directions, channels, features, domains, AdjMat):
    '''
    Args:
        directions (int): 2. The number of candidate attention directions (left or right).
        channels (int): 64. The number of EEG channels.
        features (int): 32 (KUL and SNHL datasets) or 20 (AV-GC-AAD dataset). The dimension of the differential entropy feature on each EEG channel. It depends on how many frequency bands you have.
        domains (int): 15. The number of subjects. For example, there are 15 subjects used for training in the KUL dataset.
        AdjMat (np.array): The precomputed adjacency matrix.

    Returns:
        tf.keras.Model.
            model input: EEG with the shape of (num_samples_per_subject, num_subjects, num_EEG_channels, dimension of differential entropy)
            model output: Predicted label with the shape of (num_samples_per_subject, num_subjects, 2)
                          Encoded EEG and references used for MKMMD with the shape of (num_samples_per_subject, num_subjects+2, dimension of differential entropy)
    '''
    # Input
    eeg_in = tf.keras.Input(shape=(domains, channels, features), dtype='float32')   # ([batch, domains, channels, features])
    eeg = tf.unstack(eeg_in, num=domains, axis=1)



    AdjMat = tf.tile(tf.expand_dims(AdjMat,0), (tf.shape(eeg_in)[0],1,1))   # adjacency matrix

    noise = tf.random.normal([tf.shape(eeg_in)[0], features])   # random seed

    # 1) Encoder
    Normalization = tf.keras.layers.LayerNormalization(axis=(-2,-1))
    eeg = [Normalization(eeg[i]) for i in range(domains)]

    TVGNNlayer = TVGNN(features, domains)
    encode = [TVGNNlayer([eeg[i], AdjMat]) for i in range(domains)]

    encode = tf.stack([encode[i] for i in range(domains)], axis=1) # ([batch, domains, features])

    # 2) Generator
    generated_images = generator(features)(noise)  # ([batch, features])
    generated_images1 = generator(features)(noise)

    generated_images = tf.expand_dims(generated_images, 1)  # ([batch, 1, features])
    generated_images1 = tf.expand_dims(generated_images1, 1)

    mmd = tf.keras.layers.concatenate([encode, generated_images, generated_images1], axis=1, name='MMD')    # ([batch, domains+2, features])

    # 3) Classfier
    Dense = tf.keras.layers.Dense(round((features+directions) /2),
                                  activation=squareplus, kernel_initializer='he_uniform')
    BN = tf.keras.layers.BatchNormalization()
    Softmax = tf.keras.layers.Dense(directions, activation='softmax')

    label = tf.keras.models.Sequential([Dense, BN, Softmax], name='label')(encode)    # ([batch, domains, directions])

    # Building a model
    model = tf.keras.Model(inputs=eeg_in, outputs=[label, mmd])
    model.summary(show_trainable=False)
    global w0, w1;  w0, w1 = K.variable(1.), K.variable(0.)
    model.compile(loss=['sparse_categorical_crossentropy', MK_MMD()],
                  loss_weights=[w0, w1],
                  optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=5e-3, weight_decay=5e-3,
                                                                   use_ema=True, amsgrad=True, jit_compile=True),
                  metrics=[['sparse_categorical_accuracy'], []],
                  run_eagerly=True)
    return model

# ======================== How to use it in leave-one-subject-out cross-validation ========================
directions = 2 # left / right
channels = 64
features = 32 # e.g., 32 frequency bands for the KUL dataset
domains = 15 # e.g., 15 sources subjects (16 in total, leaving one for testing)
AdjMat = np.random.normal(size=(channels, channels))    # change to the precomputed adjacency matrix!

model = AAD(directions, channels, features, domains, AdjMat)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='label_sparse_categorical_accuracy', min_delta=1e-16, patience=10,
                                                  verbose=2, mode='auto', restore_best_weights=True)
"""
EEG_train: differential entropy features for training
           the shape is (num_samples_per_subject, num_subjects, num_channels, num_freqnency_bands), e.g., (10000, 15, 64, 32)

LOC_train: ground_truth label, left is 1 and right is 2
           the shape is (num_samples_per_subject, num_subjects), e.g., (10000, 15)
"""
model.fit(EEG_train,
          [LOC_train, LOC_train],
          batch_size=32, epochs=80, verbose=1, initial_epoch=0, shuffle=True,
          callbacks=[learning_rate(80, model.optimizer.lr.numpy()), Weight(w0, w1), early_stopping])
"""
EEG_test: differential entropy features for testing
          with the shape of (num_samples, num_channels, num_freqnency_bands), e.g., (8000, 64, 32)
          Simply, the differential entropy of the target subject is replicated on axis=1 to match the dimensions of EEG_train
          the shape will be changed to  (num_samples, num_subjects, num_channels, num_freqnency_bands), e.g., (8000, 15, 64, 32)
"""
EEG_test = np.tile(np.expand_dims(EEG_test,1), (1,domains,1,1))
LOC_predict, _ = model.predict(EEG_test, verbose=0)
