# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate

        return self.learning_rate

class CosineAnnealingWithWarmRestarts(object):
    """Cosine annealing scheduler, implemented as in https://arxiv.org/pdf/1608.03983.pdf"""

    def __init__(self, min_learning_rate, max_learning_rate, total_iters_per_period, max_learning_rate_discount_factor,
                 period_iteration_expansion_factor):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_learning_rate: The minimum learning rate the scheduler can assign
        :param max_learning_rate: The maximum learning rate the scheduler can assign
        :param total_epochs_per_period: The number of epochs in a period
        :param max_learning_rate_discount_factor: The rate of discount for the maximum learning rate after each restart i.e. how many times smaller the max learning rate will be after a restart compared to the previous one
        :param period_iteration_expansion_factor: The rate of expansion of the period epochs. e.g. if it's set to 1 then all periods have the same number of epochs, if it's larger than 1 then each subsequent period will have more epochs and vice versa.
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.total_epochs_per_period = total_iters_per_period

        self.max_learning_rate_discount_factor = max_learning_rate_discount_factor
        self.period_iteration_expansion_factor = period_iteration_expansion_factor

        self.origin_total_epochs_per_period = self.total_epochs_per_period



    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """

        #calculate current period
        current_period = -1
        nextPeriodTotalNumber = 0
        current_epoch_number = 0

        while epoch_number >= nextPeriodTotalNumber:
            nextPeriodTotalNumber += self.origin_total_epochs_per_period * pow(self.period_iteration_expansion_factor,current_period)
            current_period += 1

        #calculate the index of the epoch number in the period
        previousPeriodTotalNumber = nextPeriodTotalNumber - self.origin_total_epochs_per_period * pow(self.period_iteration_expansion_factor,current_period)
        current_epoch_number = epoch_number - previousPeriodTotalNumber

        max_learning_rate = self.max_learning_rate * pow(self.max_learning_rate_discount_factor,current_period)

        i = current_epoch_number % self.total_epochs_per_period

        self.total_epochs_per_period = self.total_epochs_per_period * pow(self.period_iteration_expansion_factor,
                                                                          current_period)

        x = 0
        if i == 0:
            x = 0
        else:
            x = i / self.total_epochs_per_period * np.pi
        self.learning_rate = self.min_learning_rate + 0.5 * (max_learning_rate - self.min_learning_rate) * (1 + np.cos(x))

        learning_rule.learning_rate = self.learning_rate




        return self.learning_rate




