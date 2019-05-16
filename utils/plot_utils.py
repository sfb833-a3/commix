import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def create_plots(training_loss, reg_losses, validation_loss, num_epochs, regularization, save_path):
    epochs = np.arange(1, num_epochs, 1)
    plt.plot(epochs, training_loss,  label="training_loss", linewidth=0.7)
    plt.plot(epochs, validation_loss,   label="validation_loss", linewidth=0.7)

    if regularization:
        plt.plot(epochs, reg_losses, label="regularized_loss", linewidth=0.7)

    plt.legend(loc='upper right')
    plt.savefig(save_path)
