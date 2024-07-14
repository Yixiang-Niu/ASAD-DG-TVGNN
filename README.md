# ASAD-DG-TVGNN
Python code for "Subject-independent auditory spatial attention detection based on brain topology modeling and feature distribution alignment", submitted to _Hearing Research_

## How to use it in leave-one-subject-out cross-validation
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

Note: Since the label of each EEG segment (left/right) is unknown in advance, y_true needs to be load to compute MKMMD, and then each batch is divided into two subsets according to y_true. This requires run_eagerly=True in model.compile(), which slows down training considerably. Therefore, for all source subjects, it is advisable to preset each batch in which EEG segments with odd indices belong to one class (e.g., left) and those with even indices belong to another class (e.g., right). This does not require loading y_true and thus you can set run_eagerly=False.
