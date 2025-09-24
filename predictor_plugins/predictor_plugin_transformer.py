#!/usr/bin/env python
"""Refactored Transformer-based multi-horizon predictor plugin.

Interface preserved; internals modularized.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, Add, Conv1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K
from tqdm import tqdm
from tensorflow.keras.regularizers import l2
from .common.losses import mae_magnitude, r2_metric, composite_loss_multihead as composite_loss, random_normal_initializer_44, compute_mmd
from .common.callbacks import ReduceLROnPlateauWithCounter, EarlyStoppingWithPatienceCounter

def _get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def _positional_encoding(position, d_model):
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def _posterior_mean_field(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str): name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    c = np.log(np.expm1(1.))
    loc = tf.Variable(tf.random.normal([n], stddev=0.05, seed=42), dtype=dtype, trainable=trainable, name=(f"{name}_loc" if name else "posterior_loc"))
    scale = tf.Variable(tf.random.normal([n], stddev=0.05, seed=43), dtype=dtype, trainable=trainable, name=(f"{name}_scale" if name else "posterior_scale"))
    scale = 1e-3 + tf.nn.softplus(scale + c)
    scale = tf.clip_by_value(scale, 1e-3, 1.0)
    loc_r = tf.reshape(loc, kernel_shape)
    scale_r = tf.reshape(scale, kernel_shape)
    return tfp.distributions.Independent(tfp.distributions.Normal(loc=loc_r, scale=scale_r), reinterpreted_batch_ndims=len(kernel_shape))

def _prior_fn(dtype, kernel_shape, bias_size, trainable, name):
    if not isinstance(name, str): name = None
    bias_size = 0
    n = int(np.prod(kernel_shape)) + bias_size
    loc = tf.zeros([n], dtype=dtype)
    scale = tf.ones([n], dtype=dtype)
    loc_r = tf.reshape(loc, kernel_shape)
    scale_r = tf.reshape(scale, kernel_shape)
    return tfp.distributions.Independent(tfp.distributions.Normal(loc=loc_r, scale=scale_r), reinterpreted_batch_ndims=len(kernel_shape))

class Plugin:
    plugin_params = {
        'batch_size': 32, 'num_branch_layers': 2, 'branch_units': 32, 'merged_units': 64,
        'learning_rate': 0.0001, 'activation': 'relu', 'l2_reg': 1e-5, 'mmd_lambda': 1e-3,
        'time_horizon': 6
    }
    plugin_debug_vars = ['batch_size','num_branch_layers','branch_units','merged_units','learning_rate','l2_reg','time_horizon']

    def __init__(self, config=None):
        # Keep backward-compatible no-arg constructor; config can be supplied later to build_model/train
        self.plugin_params = {"l2_reg":0.001,"activation":"relu","branch_units":64,"merged_units":128,
                              "learning_rate":0.001,"mmd_lambda":0.1,"sigma_mmd":1.0,"predicted_horizons":[1]}
        self.params = self.plugin_params.copy()
        if config:
            self.params.update(config)
        ph = self.params.get('predicted_horizons', [1])
        n_out = len(ph)
        self.model=None; self.kl_weight_var = tf.Variable(0.0, trainable=False,dtype=tf.float32,name='kl_weight_var')
        self.local_p_control=[tf.Variable(0.0,trainable=False) for _ in range(n_out)]
        self.local_i_control=[tf.Variable(0.0,trainable=False) for _ in range(n_out)]
        self.local_d_control=[tf.Variable(0.0,trainable=False) for _ in range(n_out)]
        self.last_signed_error=[tf.Variable(0.0,trainable=False) for _ in range(n_out)]
        self.last_stddev=[tf.Variable(0.0,trainable=False) for _ in range(n_out)]
        self.last_mmd=[tf.Variable(0.0,trainable=False) for _ in range(n_out)]
        self.local_feedback=[tf.Variable(0.0,trainable=False) for _ in range(n_out)]
        if not hasattr(tfp.layers.DenseFlipout,'_already_patched_add_variable'):
            def _patched_add_variable(layer_instance,name,shape,dtype,initializer,trainable,**kwargs):
                return layer_instance.add_weight(name=name,shape=shape,dtype=dtype,initializer=initializer,trainable=trainable,**kwargs)
            tfp.layers.DenseFlipout.add_variable=_patched_add_variable; tfp.layers.DenseFlipout._already_patched_add_variable=True

    def set_params(self, **kwargs):
        for k,v in kwargs.items(): self.params[k]=v
    def get_debug_info(self): return {v:self.params[v] for v in self.plugin_debug_vars}
    def add_debug_info(self, debug_info): debug_info.update(self.get_debug_info())

    def build_model(self, input_shape, x_train, config):
        w, c = input_shape; ph = config['predicted_horizons']; l2_reg_v=config.get('l2_reg',self.params['l2_reg']); act=config.get('activation',self.params['activation'])
        merged_units=config.get('initial_layer_size',self.params['merged_units']); branch_units=merged_units//config.get('layer_size_divisor',2); lstm_units=branch_units//config.get('layer_size_divisor',2)
        inputs=Input(shape=(w,c),name='input_layer'); x=inputs
        seq_len=x.shape[1]; feat_dim=x.shape[2]; x = x + _positional_encoding(seq_len, feat_dim)
        heads=config.get('num_attention_heads',2); key_dim=feat_dim//max(1,heads)
        attn=MultiHeadAttention(num_heads=heads,key_dim=key_dim,kernel_regularizer=l2(l2_reg_v),name='mh_attention')(x,x)
        x=Add()([x,attn]); x=LayerNormalization()(x)
        x=Conv1D(filters=merged_units,kernel_size=3,strides=2,padding='same',activation=act,kernel_regularizer=l2(l2_reg_v),name='conv_1')(x)
        x=Conv1D(filters=branch_units,kernel_size=3,strides=2,padding='same',activation=act,kernel_regularizer=l2(l2_reg_v),name='conv_2')(x)
        merged=x; outputs=[]; self.output_names=[]; KL_WEIGHT=self.kl_weight_var; DenseFlipout=tfp.layers.DenseFlipout
        for horizon in ph:
            suf=f"_h{horizon}"; h_in=Conv1D(filters=branch_units,kernel_size=3,strides=2,padding='valid',kernel_regularizer=l2(l2_reg_v),name=f'head_conv1{suf}')(merged)
            h_in=Conv1D(filters=lstm_units,kernel_size=3,strides=2,padding='valid',kernel_regularizer=l2(l2_reg_v),name=f'head_conv2{suf}')(h_in)
            lstm_out=Bidirectional(LSTM(lstm_units,return_sequences=False),name=f'bilstm{suf}')(h_in)
            flip_name=f'flipout{suf}'
            flip_layer=DenseFlipout(units=1,activation='linear',kernel_posterior_fn=lambda dt,sh,bs,tr,nm=flip_name:_posterior_mean_field(dt,sh,bs,tr,nm),kernel_prior_fn=lambda dt,sh,bs,tr,nm=flip_name:_prior_fn(dt,sh,bs,tr,nm),kernel_divergence_fn=lambda q,p,_: tfp.distributions.kl_divergence(q,p)*KL_WEIGHT,name=flip_name)
            bayes=Lambda(lambda t: flip_layer(t), output_shape=lambda s:(s[0],1), name=f'bayes_out{suf}')(lstm_out)
            bias=Dense(1,activation='linear',kernel_initializer=random_normal_initializer_44,name=f'bias{suf}')(lstm_out)
            final=Add(name=f'output_horizon_{horizon}')([bayes,bias]); outputs.append(final); self.output_names.append(f'output_horizon_{horizon}')
        self.model=Model(inputs=inputs,outputs=outputs,name=f'TransformerPredictor_{len(ph)}H')
        optimizer=AdamW(learning_rate=config.get('learning_rate',self.params['learning_rate']))
        mmd_lambda=config.get('mmd_lambda',self.params['mmd_lambda']); sigma_mmd=config.get('sigma_mmd',self.params.get('sigma_mmd',1.0))
        loss_dict={}
        for i,name in enumerate(self.output_names):
            loss_dict[name]=(lambda idx=i: (lambda yt,yp: composite_loss(yt,yp,head_index=idx,mmd_lambda=mmd_lambda,sigma=sigma_mmd,p=0,i=0,d=0,list_last_signed_error=[],list_last_stddev=[],list_last_mmd=[],list_local_feedback=[])))()
        metrics_dict={n:[mae_magnitude] for n in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        self.model.summary(line_length=140)

    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        if 'predicted_horizons' not in config or 'plotted_horizon' not in config: raise ValueError("Config must have 'predicted_horizons' and 'plotted_horizon'.")
        ph=config['predicted_horizons']; plotted=config['plotted_horizon']
        if plotted not in ph: raise ValueError('plotted_horizon must be inside predicted_horizons')
        class KLAnneal(tf.keras.callbacks.Callback):
            def __init__(self, plugin, target, epochs_anneal): super().__init__(); self.plugin=plugin; self.target=target; self.epochs_anneal=epochs_anneal
            def on_epoch_begin(self, epoch, logs=None): self.plugin.kl_weight_var.assign(self.target*min(1.0,(epoch+1)/self.epochs_anneal))
        callbacks=[
            EarlyStoppingWithPatienceCounter(monitor='val_loss',patience=self.params.get('early_patience',10),restore_best_weights=True,verbose=1),
            ReduceLROnPlateauWithCounter(monitor='val_loss',factor=0.5,patience=max(1,self.params.get('early_patience',10)//4),verbose=1),
            LambdaCallback(on_epoch_end=lambda e,l: print(f"Epoch {e+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")),
            KLAnneal(self,self.params.get('kl_weight',1e-3),config.get('kl_anneal_epochs',10))
        ]
        history=self.model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_val,y_val),callbacks=callbacks,verbose=1)
        train_preds, train_unc = self.predict_with_uncertainty(x_train, config.get('mc_samples',50))
        val_preds, val_unc = self.predict_with_uncertainty(x_val, config.get('mc_samples',50))
        idx=ph.index(plotted)
        try:
            self.calculate_mae(y_train[self.output_names[idx]], train_preds[idx]); self.calculate_r2(y_train[self.output_names[idx]], train_preds[idx])
        except Exception as e: print(f'Metric calculation error: {e}')
        return history, train_preds, train_unc, val_preds, val_unc

    def predict_with_uncertainty(self, x_test, mc_samples=100):
        if self.model is None: raise ValueError('Model not built.')
        sample=self.model(x_test[:1],training=True); sample=[sample] if not isinstance(sample,list) else sample
        heads=len(sample); n=x_test.shape[0]; d=sample[0].shape[-1]
        means=[np.zeros((n,d),dtype=np.float32) for _ in range(heads)]; m2=[np.zeros_like(means[0]) for _ in range(heads)]; counts=[0]*heads
        for _ in tqdm(range(mc_samples),desc='MC'):
            preds=self.model(x_test,training=False); preds=[preds] if not isinstance(preds,list) else preds
            for h in range(heads):
                arr=preds[h].numpy(); arr=np.expand_dims(arr,-1) if arr.ndim==1 else arr
                counts[h]+=1; delta=arr-means[h]; means[h]+=delta/counts[h]; delta2=arr-means[h]; m2[h]+=delta*delta2
        stds=[]
        for h in range(heads):
            var = np.full_like(means[h], np.nan) if counts[h]<2 else m2[h]/(counts[h]-1)
            stds.append(np.sqrt(np.maximum(var,0)))
        return means, stds

    def save(self, file_path): self.model.save(file_path); print(f'Model saved to {file_path}')
    def load(self, file_path):
        self.model = load_model(file_path, custom_objects={'composite_loss': composite_loss,'mae_magnitude': mae_magnitude,'r2_metric': r2_metric})
        print(f'Predictor model loaded from {file_path}')
    def calculate_mae(self, y_true, y_pred):
        if len(y_true.shape)==1 or (len(y_true.shape)==2 and y_true.shape[1]==1): y_true=np.reshape(y_true,(-1,1)); y_true=np.concatenate([y_true,np.zeros_like(y_true)],axis=1)
        mag_true=y_true[:,0:1]; mag_pred=y_pred[:,0:1]; mae=np.mean(np.abs(mag_true.flatten()-mag_pred.flatten())); print(f'MAE (magnitude): {mae}'); return mae
    def calculate_r2(self, y_true, y_pred):
        if len(y_true.shape)==1 or (len(y_true.shape)==2 and y_true.shape[1]==1): y_true=np.reshape(y_true,(-1,1)); y_true=np.concatenate([y_true,np.zeros_like(y_true)],axis=1)
        mag_true=y_true[:,0:1]; mag_pred=y_pred[:,0:1]; ss_res=np.sum((mag_true-mag_pred)**2,axis=0); ss_tot=np.sum((mag_true-np.mean(mag_true,axis=0))**2,axis=0); r2=1-(ss_res/(ss_tot+np.finfo(float).eps)); r2=float(np.mean(r2)); print(f'R2 (magnitude): {r2}'); return r2

if __name__=='__main__':
    pass

    # --- Method within YourPredictorPlugin class ---
    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        """
        Trains the multi-output model using provided data and configuration.

        Expects y_train and y_val to be dictionaries mapping output layer names
        to their corresponding target numpy arrays (e.g., shape [num_samples, 1]).
        Utilizes KL annealing and other callbacks during training.
        Calculates final metrics based on the specific output head designated
        by config['plotted_horizon'].

        Args:
            x_train (np.ndarray): Training input features.
            y_train (dict): Dictionary of training target arrays for each output head.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            threshold_error: (Not used in provided snippet, kept for signature consistency).
            x_val (np.ndarray): Validation input features.
            y_val (dict): Dictionary of validation target arrays for each output head.
            config (dict): Configuration dictionary for training parameters, MUST contain
                           'predicted_horizons' (list) and 'plotted_horizon' (int).

        Returns:
            tuple: Contains history object, list of train predictions per head,
                   list of train uncertainties (placeholders), list of validation
                   predictions per head, list of validation uncertainties (placeholders).

        Raises:
            ValueError: If config is missing required keys or if 'plotted_horizon'
                        is not found within 'predicted_horizons'.
            TypeError: If y_train or y_val are not dictionaries.
            AttributeError: If self.output_names was not set by build_model.
        """
        # --- Configuration Validation ---
        if config is None:
            raise ValueError("Configuration dictionary ('config') is required for training.")
        if 'predicted_horizons' not in config:
            raise ValueError("Config dictionary must contain the key 'predicted_horizons' (list of ints).")
        if 'plotted_horizon' not in config:
            raise ValueError("Config dictionary must contain the key 'plotted_horizon' (int).")

        predicted_horizons = config['predicted_horizons']
        plotted_horizon = config['plotted_horizon'] # Horizon used for final metric reporting

        # Validate that the plotted_horizon is one of the predicted horizons
        if plotted_horizon not in predicted_horizons:
            raise ValueError(
                f"Invalid configuration: 'plotted_horizon' ({plotted_horizon}) "
                f"is not present in the 'predicted_horizons' list ({predicted_horizons}). "
                f"Please ensure 'plotted_horizon' matches one of the values in 'predicted_horizons'."
            )

        # Find the index corresponding to the plotted horizon
        try:
            plotted_index = predicted_horizons.index(plotted_horizon)
        except ValueError:
             # This case should be caught by the 'in' check above, but added for robustness
             raise ValueError(f"'plotted_horizon' {plotted_horizon} not found in {predicted_horizons} (index error).")


        # --- Inner Class for KL Annealing Callback ---
        class KLAnnealingCallback(tf.keras.callbacks.Callback):
            # ... (keep implementation as provided) ...
            def __init__(self, plugin, target_kl, anneal_epochs):
                super().__init__()
                self.plugin = plugin
                self.target_kl = target_kl
                self.anneal_epochs = anneal_epochs
            def on_epoch_begin(self, epoch, logs=None):
                new_kl = self.target_kl * min(1.0, (epoch + 1) / self.anneal_epochs)
                self.plugin.kl_weight_var.assign(new_kl)

        # --- Setup Callbacks ---
        anneal_epochs = config.get("kl_anneal_epochs", self.params.get("kl_anneal_epochs", 10))
        target_kl = self.params.get('kl_weight', 1e-3)
        kl_callback = KLAnnealingCallback(self, target_kl, anneal_epochs)
        min_delta_early_stopping = config.get("min_delta", self.params.get("min_delta", 1e-7))
        patience_early_stopping = self.params.get('early_patience', 10)
        start_from_epoch_es = self.params.get('start_from_epoch', 10)
        patience_reduce_lr = config.get("reduce_lr_patience", max(1, int(patience_early_stopping / 4)))

        # Instantiate callbacks WITHOUT ClearMemoryCallback
        # Assumes relevant Callback classes are imported/defined
        callbacks = [
            EarlyStoppingWithPatienceCounter(
                monitor='val_loss', patience=patience_early_stopping, restore_best_weights=True,
                verbose=1, start_from_epoch=start_from_epoch_es, min_delta=min_delta_early_stopping
            ),
            ReduceLROnPlateauWithCounter(
                monitor="val_loss", factor=0.5, patience=patience_reduce_lr, cooldown=5, min_delta=min_delta_early_stopping, verbose=1
            ),
            LambdaCallback(on_epoch_end=lambda epoch, logs:
                           print(f"Epoch {epoch+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")),
            # Removed: ClearMemoryCallback(), # <<< REMOVED THIS LINE
            kl_callback
        ]

        # --- Input Data Verification ---
        if not isinstance(y_train, dict) or not isinstance(y_val, dict):
             raise TypeError("y_train and y_val must be dictionaries.")
        if not hasattr(self, 'output_names') or not self.output_names:
             raise AttributeError("self.output_names not set by build_model.")
        plotted_output_name = f"output_horizon_{plotted_horizon}"
        if plotted_output_name not in y_train or plotted_output_name not in y_val:
             raise ValueError(f"Target dicts missing key: '{plotted_output_name}'")
        # Optional: Check all keys match
        if set(y_train.keys()) != set(self.output_names) or set(y_val.keys()) != set(self.output_names):
             print("WARN: Target data dictionary keys may not perfectly match all model output names.")

        # --- Model Training ---
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_val, y_val),
                                 callbacks=callbacks,
                                 verbose=1)

        # --- Post-Training Predictions ---
        # Note: Predicting on full train/val sets uses memory. Consider alternatives if needed.
        #list_train_preds = self.model.predict(x_train, batch_size=batch_size)
        #list_val_preds = self.model.predict(x_val, batch_size=batch_size)
        mc_samples = config.get("mc_samples", 100)
        list_train_preds,list_train_uncertainty  = self.predict_with_uncertainty(x_train, mc_samples)
        list_val_preds, list_val_uncertainty = self.predict_with_uncertainty(x_val, mc_samples)   

        # Placeholder uncertainties (as these weren't generated during training)
        #list_train_uncertainty = [np.zeros_like(preds) for preds in list_train_preds]
        #list_val_uncertainty = [np.zeros_like(preds) for preds in list_val_preds]

        # --- Post-Training Metrics (for the configured 'plotted_horizon') ---
        # Assumes self.calculate_mae and self.calculate_r2 methods exist
        try:
            y_train_plotted = y_train[plotted_output_name]
            train_preds_plotted = list_train_preds[plotted_index] # Use pre-calculated index
            print(f"Calculating final MAE/R2 for plotted horizon: {plotted_horizon} (Index: {plotted_index})")
            if hasattr(self, 'calculate_mae') and callable(self.calculate_mae):
                self.calculate_mae(y_train_plotted, train_preds_plotted)
            if hasattr(self, 'calculate_r2') and callable(self.calculate_r2):
                self.calculate_r2(y_train_plotted, train_preds_plotted)
        except Exception as e:
             print(f"ERROR during post-training metric calculation: {e}")

        # Return history and lists of predictions/uncertainties
        return history, list_train_preds, list_train_uncertainty, list_val_preds, list_val_uncertainty
    

    # --- Method within PredictorPluginANN class ---
    # --- Method within PredictorPluginANN class ---
    def predict_with_uncertainty(self, x_test, mc_samples=100):
        """
        Performs Monte Carlo dropout predictions for the multi-output model
        using an incremental approach to avoid large memory allocation.

        Runs the model multiple times with dropout enabled (training=True)
        to estimate predictive uncertainty (standard deviation) for each output head.

        Args:
            x_test (np.ndarray): Input data for prediction.
            mc_samples (int): Number of Monte Carlo samples to perform.

        Returns:
            tuple: (list_mean_predictions, list_uncertainty_estimates)
                   Lists containing numpy arrays (one per output head)
                   for mean predictions and standard deviations (uncertainty).
                   Shape of each array: [num_samples, output_dim (usually 1)].
        """
        if self.model is None:
            raise ValueError("Model has not been built or loaded.")
        if mc_samples <= 0:
            return [], []

        # Get dimensions from a single sample run
        try:
            first_run_output_tf = self.model(x_test[:1], training=True) # Predict on one sample
            if not isinstance(first_run_output_tf, list): first_run_output_tf = [first_run_output_tf]
            num_heads = len(first_run_output_tf)
            if num_heads == 0: return [], []
            first_head_output = first_run_output_tf[0].numpy()
            num_test_samples = x_test.shape[0]
            output_dim = first_head_output.shape[1] if first_head_output.ndim > 1 else 1
        except Exception as e:
            print(f"ERROR getting model output shape in predict_with_uncertainty: {e}")
            raise ValueError("Could not determine model output structure.") from e

        # Initialize accumulators for mean and variance calculation (Welford's algorithm components)
        # Using lists to store per-head accumulators
        means = [np.zeros((num_test_samples, output_dim), dtype=np.float32) for _ in range(num_heads)]
        m2s = [np.zeros((num_test_samples, output_dim), dtype=np.float32) for _ in range(num_heads)]
        counts = [0] * num_heads # Use a single count across heads, assuming samples are drawn together

        # print(f"Running {mc_samples} MC samples for uncertainty (incremental)...") # Informative print
        for i in tqdm(range(mc_samples), desc="MC Samples"):
            # Get predictions for all heads in this sample
            batch_size = 256  # ✅ Use safe batch size
            ## Initialize a list for each output head
            head_outputs_lists = None
            for i in range(0, len(x_test), batch_size):
                batch_x = x_test[i:i + batch_size]
                preds = self.model(batch_x, training=False)
                if not isinstance(preds, list):
                    preds = [preds]
                if head_outputs_lists is None:
                    head_outputs_lists = [[] for _ in range(len(preds))]
                for h, pred in enumerate(preds):
                    head_outputs_lists[h].append(pred)


            # Concatenate outputs for each head along the batch dimension
            head_outputs_tf = [tf.concat(head_list, axis=0) for head_list in head_outputs_lists]



            if not isinstance(head_outputs_tf, list): head_outputs_tf = [head_outputs_tf]

            # Process each head's output for this sample
            for h in range(num_heads):
                head_output_np = head_outputs_tf[h].numpy()
                # Reshape if necessary
                if head_output_np.ndim == 1:
                    head_output_np = np.expand_dims(head_output_np, axis=-1)
                if head_output_np.shape != (num_test_samples, output_dim):
                     raise ValueError(f"Shape mismatch in MC sample {i}, head {h}: Expected {(num_test_samples, output_dim)}, got {head_output_np.shape}")

                # Welford's online algorithm update
                counts[h] += 1
                delta = head_output_np - means[h]
                means[h] += delta / counts[h]
                delta2 = head_output_np - means[h] # New delta using updated mean
                m2s[h] += delta * delta2

            # Optional progress print
            # if (i + 1) % (mc_samples // 10 or 1) == 0: print(f"  MC sample {i+1}/{mc_samples}")

        # Finalize calculations: variance = M2 / (n - 1), stddev = sqrt(variance)
        list_mean_predictions = means # The mean is already calculated
        list_uncertainty_estimates = []
        for h in range(num_heads):
             if counts[h] < 2: # Need at least 2 samples for variance/stddev
                 variance = np.full((num_test_samples, output_dim), np.nan, dtype=np.float32)
             else:
                 variance = m2s[h] / (counts[h] - 1)
             stddev = np.sqrt(np.maximum(variance, 0)) # Ensure variance isn't negative due to float issues
             list_uncertainty_estimates.append(stddev.astype(np.float32))

        # print("MC sampling finished.") # Informative print
        return list_mean_predictions, list_uncertainty_estimates
    
    
    def save(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        self.model = load_model(file_path, custom_objects={
            "composite_loss": composite_loss,
            "compute_mmd": compute_mmd,
            "r2_metric": r2_metric,
            "mae_magnitude": mae_magnitude
        })
        print(f"Predictor model loaded from {file_path}")

    def calculate_mae(self, y_true, y_pred):
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        print(f"DEBUG: y_true (sample): {mag_true.flatten()[:5]}")
        print(f"DEBUG: y_pred (sample): {mag_pred.flatten()[:5]}")
        mae = np.mean(np.abs(mag_true.flatten() - mag_pred.flatten()))
        print(f"Calculated MAE (magnitude): {mae}")
        return mae

    def calculate_r2(self, y_true, y_pred):
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            y_true = np.reshape(y_true, (-1, 1))
            y_true = np.concatenate([y_true, np.zeros_like(y_true)], axis=1)
        mag_true = y_true[:, 0:1]
        mag_pred = y_pred[:, 0:1]
        print(f"Calculating R²: y_true shape={mag_true.shape}, y_pred shape={mag_pred.shape}")
        SS_res = np.sum((mag_true - mag_pred) ** 2, axis=0)
        SS_tot = np.sum((mag_true - np.mean(mag_true, axis=0)) ** 2, axis=0)
        r2_scores = 1 - (SS_res / (SS_tot + np.finfo(float).eps))
        r2 = np.mean(r2_scores)
        print(f"Calculated R² (magnitude): {r2}")
        return r2

# ---------------------------
# Debugging usage example (if run as main)
# ---------------------------
if __name__ == "__main__":
    plugin = Plugin()
    # For debugging, assume input shape (window_size, num_channels) where num_channels=3.
    # Example: window_size=24, 3 channels (trend, seasonal, noise).
    plugin.build_model(input_shape=(24, 3), x_train=None, config={})
    debug_info = plugin.get_debug_info()
    print(f"Debug Info: {debug_info}")