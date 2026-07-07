import numpy as np

class CuriosityEngine:
    """
    Oudeyer's Learning Progress (LP) Intrinsic Curiosity Engine.
    
    Rather than rewarding raw prediction error (which gets stuck in the "noisy television" trap),
    this engine rewards the first derivative of prediction error (learning progress).
    
    Uses an Exponential Moving Average (EMA) to filter out high-frequency sensor/control noise.
    """
    def __init__(self, window_size=50, curiosity_gain=0.1):
        self.window_size = window_size
        self.gain = curiosity_gain
        self.error_history = []
        self.ema_error = None
        self.alpha_ema = 0.05  # Noise-dampening filter constant

    def update_error(self, current_error):
        """Accumulates prediction error, applying a noise-dampening EMA filter."""
        if self.ema_error is None:
            self.ema_error = float(current_error)
        else:
            self.ema_error = self.alpha_ema * float(current_error) + (1.0 - self.alpha_ema) * self.ema_error
            
        self.error_history.append(self.ema_error)
        if len(self.error_history) > self.window_size * 2:
            self.error_history.pop(0)

    def compute_learning_progress(self):
        """Computes the decrease in prediction error over sliding window blocks."""
        if len(self.error_history) < self.window_size * 2:
            return 0.0

        # Split sliding window into old and new halves
        old_half = self.error_history[:self.window_size]
        new_half = self.error_history[self.window_size:]

        mean_old_error = np.mean(old_half)
        mean_new_error = np.mean(new_half)

        # Learning progress is old error minus new error (positive when error decreases)
        lp = mean_old_error - mean_new_error

        # Reward only positive learning progress
        return max(0.0, lp)

    def compute_intrinsic_reward(self, current_error):
        """Steps the curiosity trace and returns the scaled learning progress reward."""
        self.update_error(current_error)
        lp = self.compute_learning_progress()
        return lp * self.gain
