class Smoother:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.prev_x = None
        self.prev_y = None

    def smooth(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y

        smooth_x = self.alpha * self.prev_x + (1 - self.alpha) * x
        smooth_y = self.alpha * self.prev_y + (1 - self.alpha) * y

        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y
