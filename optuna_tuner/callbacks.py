import optuna

class ProgressCallback:

    def __init__(self, n_trials: int, verbose: bool = True):
        self.n_trials = n_trials
        self.verbose = verbose

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if not self.verbose:
            return
        t = trial.number + 1
        print(
            f"  Trial {t:>4}/{self.n_trials} "
            f"| Score: {trial.value:.6f} "
            f"| Mejor: {study.best_value:.6f}"
        )