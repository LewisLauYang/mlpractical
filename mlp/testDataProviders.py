

import numpy as np
import matplotlib.pyplot as plt
import mlp.data_providers as data_providers

mnist_dp = data_providers.MNISTDataProvider(
    which_set='valid', batch_size=5, max_num_batches=5, shuffle_order=False)

for inputs, targets in mnist_dp:
    assert np.all(targets.sum(-1) == 1.)
    assert np.all(targets >= 0.)
    assert np.all(targets <= 1.)
    print(targets)


