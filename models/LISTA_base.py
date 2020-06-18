#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file  : LISTA_base.py
author: xhchrn
email : chernxh@tamu.edu
date  : 2019-02-18

A base class for all LISTA networks.
"""

import numpy as np
import numpy.linalg as la
import tensorflow as tf
import sys, os
import time

import utils.train

class LISTA_base (object):

    """
    Implementation of deep neural network model.
    """

    def __init__ (self):
        pass

    def setup_layers (self):
        pass

    def inference (self):
        pass

    def save_trainable_variables (self , filename, **kwargs):
        """
        Save trainable variables in the model to npz file with current value of each
        variable in tf.trainable_variables().

        :sess: Tensorflow session.
        :savefn: File name of saved file.

        """
        state = getattr (self , 'state' , {})

        """
        Save trainable variables in the model to npz file with current value of
        each variable in tf.trainable_variables().

        :sess: Tensorflow session.
        :filename: File name of saved file.
        :scope: Name of the variable scope that we want to save.
        :kwargs: Other arguments that we want to save.

        """
        save = dict()
        for v_tuple  in self.vars_in_layer:
            for v in v_tuple:
                save[str(v.name)] = v

        # file name suffix check
        if not filename.endswith(".npz"):
            filename = filename + ".npz"

        save.update(self._scope)
        save.update(state)

        np.savez(filename, **save)

    def load_trainable_variables (self, filename):
        """
        Load trainable variables from saved file.

        :sess: TODO
        :savefn: TODO
        :returns: TODO

        """
        """
        Load trainable variables from saved file.

        :sess: TODO
        :filename: TODO
        :returns: TODO

        """
        other = dict()
        # file name suffix check
        if filename[-4:] != '.npz':
            filename = filename + '.npz'
        if not os.path.exists(filename):
            raise ValueError(filename + ' not exists')

        tv = dict([(str(v.name), v) for v_tuple in self.vars_in_layer for v in v_tuple ])
        for k, d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                tf.compat.v1.assign(tv[k], d)
            else:
                other[k] = d

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

