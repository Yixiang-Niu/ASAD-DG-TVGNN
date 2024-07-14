# ASAD-DG-TVGNN
Python code for "Subject-independent auditory spatial attention detection based on brain topology modeling and feature distribution alignment", submitted to _Hearing Research_

Note: Since the label of each EEG segment (left/right) is unknown in advance, y_true needs to be load to compute MKMMD, and then each batch is divided into two subsets according to y_true. This requires run_eagerly=True in model.compile(), which slows down training considerably. Therefore, for all source subjects, it is advisable to preset each batch in which EEG segments with odd indices belong to one class (e.g., left) and those with even indices belong to another class (e.g., right). This does not require loading y_true and thus you can set run_eagerly=False.
