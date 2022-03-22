import tensorflow as tf


class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    """ 
    Overload from keras EarlyStopping
    
    Start "monitor" after designated iteration counts
    
    Attributes:
        min_epoch (int): minimum epoch to initiate the callback function
    """
    def __init__(self, monitor="val_loss", min_delta=0, patience=0, verbose=0, mode="auto", min_epoch=0, baseline=None, restore_best_weights=False):
        super(EarlyStopping, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
        self.min_epoch = min_epoch
    
    def on_epoch_end(self, epoch, logs):
        if epoch < self.min_epoch:
            return
        return super().on_epoch_end(epoch, logs=logs)


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """  
    Overload from keras ModelCheckpoint
    
    Start "monitor" after designated iteration counts
    
    Attributes:
        min_epoch (int): minimum epoch to initiate the callback function 
    """
    def __init__(self, filepath, monitor="val_loss", verbose=0, min_epoch=0, save_best_only=False, save_weights_only=False, mode="auto", save_freq="epoch", options=None, **kwargs):
        super().__init__(filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only, mode=mode, save_freq=save_freq, options=options, **kwargs)
        self.min_epoch = min_epoch
    
    def on_epoch_end(self, epoch, logs):
        if epoch < self.min_epoch:
            return
        return super().on_epoch_end(epoch, logs=logs)
    