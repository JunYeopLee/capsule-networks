class Config:
    batch_size = 128
    learning_rate = 1e-3
    logdir = 'gs://weighty-actor-177008-mlengine/capsNet_11_dr/logdir'
    dataset = 'gs://weighty-actor-177008-mlengine/data/data/mnist/npy'
    # logdir = 'logdir/logdir'
    # dataset = 'data/mnist'
    num_epochs = 500
