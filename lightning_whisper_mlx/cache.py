"""Request-owned attention storage with bounded capacity and explicit positions."""

import mlx.core as mx


class KVCache:
    """Store B,T,D keys/values without reallocating valid history every token."""

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("cache capacity must be positive")
        self.max_size = max_size
        self.offset = 0
        self.keys = None
        self.values = None

    def update(self, keys, values):
        end = self.offset + keys.shape[1]
        if keys.shape != values.shape or end > self.max_size:
            raise ValueError(
                "attention cache update exceeds context or has invalid shape"
            )
        if self.keys is None:
            capacity = min(self.max_size, ((end + 63) // 64) * 64)
            shape = (keys.shape[0], capacity, keys.shape[2])
            self.keys = mx.zeros(shape, dtype=keys.dtype)
            self.values = mx.zeros(shape, dtype=values.dtype)
        elif (keys.shape[0], keys.shape[2], keys.dtype) != (
            self.keys.shape[0],
            self.keys.shape[2],
            self.keys.dtype,
        ):
            raise ValueError("attention cache batch, width and dtype must remain fixed")
        self.ensure_capacity(end)
        self.keys[:, self.offset : end] = keys
        self.values[:, self.offset : end] = values
        self.offset = end
        return self.keys[:, :end], self.values[:, :end]

    def ensure_capacity(self, end):
        if self.keys is None or not self.offset <= end <= self.max_size:
            raise ValueError(
                "cache capacity request is outside the initialized context"
            )
        if end > self.keys.shape[1]:
            capacity = min(self.max_size, ((end + 63) // 64) * 64)
            padding = [(0, 0), (0, capacity - self.keys.shape[1]), (0, 0)]
            self.keys, self.values = mx.pad(self.keys, padding), mx.pad(
                self.values, padding
            )

    def truncate(self, length):
        if not 0 <= length <= self.offset:
            raise ValueError("cache truncation length is outside the current cache")
        self.offset = length

    def reorder(self, indices):
        if self.keys is not None:
            self.keys = self.keys[indices]
            self.values = self.values[indices]

    @property
    def state(self):
        """Arrays to evaluate explicitly without exposing unused positions to attention."""
        return (self.keys, self.values)

    def __iter__(self):
        yield self.keys[:, : self.offset]
        yield self.values[:, : self.offset]
