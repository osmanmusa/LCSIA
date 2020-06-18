import utils.train as train

class DataSet:

    def __init__(self, config, problem):

        self.config = config
        self. problem = problem

        self.update()

    def update(self):
        if not self.config.test:
            y_, x_, y_val_, x_val_ = (
                train.setup_input_sc(
                    self.config.test, self.problem, self.config.tbs, self.config.vbs, self.config.fixval,
                    self.config.supp_prob, self.config.SNR, self.config.magdist, **self.config.distargs))

            self.x_ = x_
            self.y_ = y_
            self.x_val_ = x_val_
            self.y_val_ = y_val_
        else:
            y_, x_ = (
                train.setup_input_sc(
                    self.config.test, self.problem, self.config.tbs, self.config.vbs, self.config.fixval,
                    self.config.supp_prob, self.config.SNR, self.config.magdist, **self.config.distargs))

            self.x_ = x_
            self.y_ = y_