class EarlyStopping:
    """
    Early stopping basé sur une métrique de validation.

    Args:
        patience (int): Nombre d'époques sans amélioration.
        min_delta (float): Amélioration minimale.
    """

    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop
