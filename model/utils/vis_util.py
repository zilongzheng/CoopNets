import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

class Visualizer():

    num_fig = 0

    def __init__(self, title=None, xlabel='epoch', ylabel=None, ylim=None, save_figpath=None, show_avg=True, avg_period=None):
        self.title = title
        self.save_figpath = save_figpath
        self.show_avg = show_avg
        self.avg_period = avg_period

        if ylabel == None:
            self.ylabel = title
        else:
            self.ylabel = ylabel

        self.xlabel = xlabel

        self.fig = None

        self.loss_vals = OrderedDict()

        plt.ion()

        self.fig = plt.figure()

        if self.show_avg:
            assert self.avg_period != None
            self.last_ys = []
            self.sma = OrderedDict()
        if show_avg:
            self.alpha = 0.3
        else:
            self.alpha = 1.0

        plt.title(self.title, fontsize=16)
        plt.xlabel(self.xlabel, fontsize=12)
        plt.ylabel(self.ylabel, fontsize=12)
        if ylim:
            plt.ylim(ylim)

    def add_loss_val(self, epoch, loss):
        self.loss_vals[epoch] = loss

        if self.show_avg:
            self.last_ys.append(loss)
            if len(self.last_ys) > self.avg_period:
                self.last_ys.pop(0)
            self.sma[epoch] = np.mean(self.last_ys)

    def draw_figure(self):
        plt.figure(self.fig.number)

        plt.plot(self.loss_vals.keys(), self.loss_vals.values(), 'r-', alpha = self.alpha)
        if self.show_avg:
            plt.plot(self.sma.keys(), self.sma.values(), 'r-')

        plt.draw()

        if self.save_figpath != None:
            self.fig.savefig(self.save_figpath, bbox_inches='tight')
