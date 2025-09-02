import time

'''
O------------------------------------------------------------------------------O
| CLOCK CLASS                                                                  |
O------------------------------------------------------------------------------O
'''

class Clock:

    def __init__(self):
        self._current_time = time.time()
        self._previous_time = self._current_time
        self._frame_time = 1.0
        self._delta_time = 0.0
        self._update_interval = 0.2
        self._num_frames = 0

    @property
    def frame_time(self):
        return self._frame_time

    @property
    def frame_rate(self):
        return 1.0 / self._frame_time

    @property
    def update_interval(self):
        return self._update_interval

    @update_interval.setter
    def update_interval(self, update_interval: float):
        self._update_interval = float(update_interval)

    def update(self):
        self._num_frames += 1
        self._current_time = time.time()
        self._delta_time += self._current_time - self._previous_time
        self._previous_time = self._current_time
        if self._delta_time >= self._update_interval:
            self._frame_time = self._delta_time / self._num_frames
            self._delta_time = 0.0
            self._num_frames = 0
