import tensorflow as tf
import utils.shrinkage as shrinkage
import numpy as np
import sys, os
import time


class LITA(tf.keras.Model):

    def __init__(self, A, T, shrink_name, untied, coord, scope):
        """
        :A      : Instance of Problem class, describing problem settings.
        :T      : Number of layers (depth) of this LISTA model.
        :lam    : Initial value of thresholds of shrinkage functions.
        :untied : Whether weights are shared within layers.
        :coord  :
        :scope  :
        """
        super(LITA, self).__init__()

        self._A   = A.astype (np.float32)
        self._T   = T
        self._M   = self._A.shape [0]
        self._N   = self._A.shape [1]

        self.denoise, self._theta = shrinkage.get_shrinkage_function(shrink_name)

        # if coord:
        #     self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta

        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2

        self._untied = untied
        self._coord  = coord
        self._scope  = scope

        """ Set up layers."""
        self.setup_layers()

        self.save_var = tf.Variable((1.0,1.0), name='save-var')

    def setup_layers(self):
        """
        Implementation of LISTA model proposed by LeCun in 2010.

        :prob: Problem setting.
        :T: Number of layers in LISTA.
        :returns:
            :layers: List of tuples ( name, xh_, var_list )
                :name: description of layers.
                :xh: estimation of sparse code at current layer.
                :var_list: list of variables to be trained seperately.

        """
        Bs_   = []
        thetas_ = []

        B = (np.transpose (self._A) / self._scale).astype (np.float32)

        with tf.compat.v1.variable_scope(self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)

            if not self._untied: # tied model
                # Bs_.append (tf.compat.v1.get_variable (name='B', dtype=tf.float32,
                #                              initializer=B))
                Bs_.append(tf.Variable(name='B', dtype=tf.float32, initial_value=B))
                Bs_ = Bs_ * self._T

            for t in range (self._T):
                # thetas_.append (tf.compat.v1.get_variable (name="theta_%d"%(t+1),
                #                                dtype=tf.float32,
                #                                initializer=self._theta))

                theta = tf.Variable(name="theta_%d" % (t + 1),
                            dtype=tf.float32,
                            initial_value=self._theta)
                thetas_.append (theta)

                # thetas_.append (tf.Variable (name="theta_%d"%(t+1),
                #                                dtype=tf.float32,
                #                                initial_value=self._theta))
                if self._untied: # untied model
                    # Bs_.append (tf.compat.v1.get_variable (name='B_%d'%(t+1),
                    #                              dtype=tf.float32,
                    #                              initializer=B))
                    Bs_.append(tf.Variable(name='B_%d' % (t + 1), dtype=tf.float32, initial_value=B))

        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (Bs_, thetas_))

    def call (self, y_, x0_=None, return_recon=False):
        xhs_  = [] # collection of the regressed sparse codes
        if return_recon:
            yhs_  = [] # collection of the reconstructed signals

        # initialization
        batch_size = tf.shape(y_)[-1]
        dxdr_ = tf.zeros (shape=(1, batch_size), dtype=tf.float32)
        if x0_ is None:
            xh_ = tf.zeros (shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append (xh_)

        OneOverM = tf.constant (float(1)/self._M, dtype=tf.float32)
        NOverM   = tf.constant (float(self._N)/self._M, dtype=tf.float32)
        vt_ = tf.zeros_like (y_, dtype=tf.float32)

        with tf.compat.v1.variable_scope(self._scope, reuse=True) as vs:
            for t in range (self._T):
                B_, theta_ = self.vars_in_layer [t]

                yh_ = tf.matmul (self._kA_, xh_)
                if return_recon:
                    yhs_.append (yh_)

                bt_   = dxdr_ * NOverM

                vt_   = y_ - yh_ + bt_ * vt_
                rvar_ = tf.reduce_sum (tf.square (vt_), axis=0) * OneOverM
                rh_ = xh_ + tf.matmul(B_, vt_)

                (xh_,dxdr_)  = self.denoise(rh_, rvar_, theta_)
                xhs_.append (xh_)

            if return_recon:
                yhs_.append (tf.matmul (self._kA_, xh_))
                return xhs_, yhs_
            else:
                return xhs_

    def do_training(self, stages, data_set, savefn, scope,
                    val_step, maxit, better_wait):
        """
        Do training actually. Refer to utils/train.py.

        :sess       : Tensorflow session, in which we will run the training.
        :stages     : List of tuples. Training stages obtained via
            `utils.train.setup_training`.
        :savefn     : String. Path where the trained model is saved.
        :batch_size : Integer. Training batch size.
        :val_step   : Integer. How many steps between two validation.
        :maxit      : Integer. Max number of iterations in each training stage.
        :better_wait: Integer. Jump to next stage if no better performance after
            certain # of iterations.

        """
        """
        Train the model actually.

        :sess: Tensorflow session. Variables should be initialized or loaded from trained
               model in this session.
        :stages: Training stages info. ( name, xh_, loss_, nmse_, op_, var_list ).
        :prob: Problem instance.
        :batch_size: Batch size.
        :val_step: How many steps between two validation.
        :maxit: Max number of iterations in each training stage.
        :better_wait: Jump to next training stage in advance if nmse_ no better after
                      certain number of steps.
        :done: name of stages that has been done.

        """
        if not savefn.endswith(".npz"):
            savefn += ".npz"
        if os.path.exists(savefn):
            sys.stdout.write('Pretrained model found. Loading...\n')
            state = self.load_trainable_variables(savefn)
        else:
            state = {}

        done = state.get('done', [])
        log = state.get('log', [])

        for name, loss_, nmse_, loss_val_, nmse_val_, opt_, var_list in stages:
            start = time.time()
            """Skip stage done already."""
            if name in done:
                sys.stdout.write('Already did {}. Skipping\n'.format(name))
                continue

            # print stage information
            var_disc = 'fine tuning ' + ','.join([v.name for v in var_list])
            print(name + ' ' + var_disc)

            nmse_hist_val = []
            for i in range(maxit + 1):

                data_set.update()
                # _, loss_tr, nmse_tr = sess.run ([op_, loss_, nmse_])
                opt_.minimize(loss=loss_, var_list=var_list)
                nmse_tr_dB = 10. * np.log10(nmse_())
                loss_tr = loss_()

                if i % val_step == 0:
                    nmse_val = nmse_val_()
                    loss_val = loss_val_()

                    if np.isnan(nmse_val):
                        raise RuntimeError('nmse is nan. exiting...')

                    nmse_hist_val = np.append(nmse_hist_val, nmse_val)
                    db_best_val = 10. * np.log10(nmse_hist_val.min())
                    nmse_val_dB = 10. * np.log10(nmse_val)
                    sys.stdout.write("\r| i={i:<7d} | loss_tr={loss_tr:.6f} | "
                                     "nmse_tr/dB={nmse_tr_db:.6f} | loss_val ={loss_val:.6f} | "
                                     "nmse_val/dB={nmse_val_db:.6f} | (best={db_best_val:.6f})" \
                                     .format(i=i, loss_tr=loss_tr, nmse_tr_db=nmse_tr_dB,
                                             loss_val=loss_val, nmse_val_db=nmse_val_dB,
                                             db_best_val=db_best_val))
                    sys.stdout.flush()
                    if i % (10 * val_step) == 0:
                        age_of_best = (len(nmse_hist_val) -
                                       nmse_hist_val.argmin() - 1)
                        # If nmse has not improved for a long time, jump to the
                        # next training stage.
                        if age_of_best * val_step > better_wait:
                            print('')
                            break
                    if i % (100 * val_step) == 0:
                        print('')

            done = np.append(done, name)
            # TODO: add log

            end = time.time()
            time_log = 'Took me {totaltime:.3f} minutes, or {time_per_interation:.1f} ms per iteration'.format(
                totaltime=(end - start) / 60, time_per_interation=(end - start) * 1000 / i)
            print(time_log)

            state['done'] = done
            state['log'] = log

            self.save_trainable_variables(savefn, **state)