import optuna

class ProgressCallback:

    def __init__(self, n_trials: int, offset: int = 0, verbose: bool = True):
        self.n_trials = n_trials
        self.offset   = offset
        self.verbose  = verbose

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if not self.verbose:
            return
        if trial.value is None:
            return
        t = self.offset + trial.number + 1
        print(
            f"  Trial {t:>4}/{self.n_trials} "
            f"| Score: {trial.value:.6f} "
            f"| Best: {study.best_value:.6f}"
        )