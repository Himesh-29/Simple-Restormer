import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingRestartCyclicLR(_LRScheduler):
    def __init__(self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert len(self.periods) == len(self.restart_weights), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = self._get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

    def _get_position_from_periods(self, iteration, cumulative_period):
        for i, period in enumerate(cumulative_period):
            if iteration <= period:
                return i
        return len(cumulative_period) - 1
