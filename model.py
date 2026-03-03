import torch
import torch.nn as nn
import numpy as np


def BoaConstrictor(d_model=256, num_layers=4, vocab_size=256, device="cpu"):
    """
    Drop-in replacement for the Mamba-based BoaConstrictor.
    Uses a multi-layer LSTM backbone — pure PyTorch, no CUDA extensions required.
    Identical external interface: forward(), init_stream(), step().
    """

    class LSTMBlock(nn.Module):
        """Single LSTM layer wrapped with LayerNorm + residual + feedforward."""
        def __init__(self, d_model: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,   # expects [B, L, D]
            )
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        def forward(self, x, hx=None):
            """
            x   : [B, L, D]
            hx  : (h, c) each [1, B, D], or None
            returns: y [B, L, D], (h, c)
            """
            y, hx_out = self.lstm(self.ln1(x), hx)
            y = self.ln2(y)
            y = self.ff(y)
            return x + y, hx_out  # residual connection

        def step(self, x_t, hx):
            """
            Single-token step for streaming inference.
            x_t : [B, D]
            hx  : (h, c) each [1, B, D]
            returns: y_t [B, D], (h, c)
            """
            x_in = self.ln1(x_t).unsqueeze(1)          # [B, 1, D]
            y, hx_out = self.lstm(x_in, hx)
            y = y.squeeze(1)                            # [B, D]
            y = self.ln2(y)
            y = self.ff(y)
            return x_t + y, hx_out

    class BoaBytePredictor(nn.Module):
        """
        LSTM model that predicts the next byte in a sequence.
        Replaces the Mamba backbone with the identical external interface:
          - forward(x, inference_params=None) -> [B, L, vocab_size]
          - init_stream(max_len, batch_size, device, dtype) -> caches
          - step(byte_t, caches) -> [B, vocab_size]
        """
        def __init__(self, d_model: int, num_layers: int, vocab_size: int):
            super().__init__()
            self.d_model = d_model
            self.num_layers = num_layers

            self.embedding = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList(
                [LSTMBlock(d_model) for _ in range(num_layers)]
            )
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, vocab_size),
            )

        def forward(self, x, inference_params=None):
            """
            x : [B, L]  long tensor of byte indices
            returns logits : [B, L, vocab_size]
            """
            h = self.embedding(x)           # [B, L, D]
            for blk in self.blocks:
                h, _ = blk(h)
            return self.head(h)             # [B, L, vocab_size]

        @torch.inference_mode()
        def init_stream(self, max_len: int, batch_size: int = 1,
                        device=None, dtype=None):
            """Returns list of (h, c) per block for streaming inference."""
            if device is None:
                device = next(self.parameters()).device
            caches = []
            for _ in self.blocks:
                h = torch.zeros(1, batch_size, self.d_model, device=device)
                c = torch.zeros(1, batch_size, self.d_model, device=device)
                caches.append((h, c))
            return caches

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
            """
            Single-token autoregressive step — O(1) per token.
            byte_t : [B]
            caches : list of (h, c) per block
            returns logits : [B, vocab_size]
            """
            h = self.embedding(byte_t)      # [B, D]
            for i, blk in enumerate(self.blocks):
                h, caches[i] = blk.step(h, caches[i])
            return self.head(h)             # [B, vocab_size]

    model = BoaBytePredictor(
        d_model=d_model,
        num_layers=num_layers,
        vocab_size=vocab_size,
    )
    return model

def _aligned_len(n_bytes: int, seq_len: int, batch_size: int) -> int:
    block = seq_len * batch_size
    return (n_bytes // block) * block


def make_splits(data_bytes, seq_len: int, batch_size: int,
                splits=(0.8, 0.1, 0.1)):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    buf = np.frombuffer(bytes(data_bytes), dtype=np.uint8)
    usable = _aligned_len(len(buf), seq_len, batch_size)
    buf = buf[:usable]

    n = len(buf)
    n_train = _aligned_len(int(n * splits[0]), seq_len, batch_size)
    n_val   = _aligned_len(int(n * splits[1]), seq_len, batch_size)
    n_test  = _aligned_len(n - n_train - n_val, seq_len, batch_size)

    i0, i1, i2 = 0, n_train, n_train + n_val
    return (
        buf[i0:i1].tobytes(),
        buf[i1:i2].tobytes(),
        buf[i2:i2 + n_test].tobytes(),
    )


class ByteDataloader:
    """Simple dataloader that yields batches of bytes."""
    def __init__(self, data_bytes, seq_len=1048576, batch_size=1, device="cpu"):
        self.data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.pos        = 0
        self.device     = device

    def __len__(self):
        return len(self.data_bytes) // (self.seq_len * self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        end = self.pos + self.seq_len * self.batch_size
        if end > len(self.data_bytes):
            self.pos = 0
            raise StopIteration
        batch_indices = (
            np.arange(self.pos, end)
              .reshape(self.batch_size, self.seq_len)
        )
        self.pos += self.seq_len * self.batch_size
        batch = self.data_bytes[batch_indices]
        return torch.tensor(batch, dtype=torch.long, device=self.device)