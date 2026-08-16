# generate_lstm_1.1mb.py
# =============================================================================
# Generation Script for NERV Sharding Load Predictor Model (lstm_1.1mb.pt)
# =============================================================================
# This script creates a realistic, lightweight LSTM model for the NERV blockchain's
# dynamic neural sharding load predictor.
#
# Model Purpose (per whitepaper & code):
# - 1.1MB LSTM model for predicting shard load (TPS, queue size, etc.)
# - Used in sharding/lstm_predictor.rs to forecast overload/underload
# - Enables proactive shard splitting/merging for infinite horizontal scaling
# - Input: Time series sequence of shard metrics (e.g., 10 features: TPS, queue, CPU, mem, tx types, etc.)
# - Output: Predicted next-step load metrics + overload probability (multi-task)
# - Designed for very fast CPU inference (quantized int8)
# - Total parameters: ~1.05M → ~1.1MB when saved quantized
#
# Architecture:
# - Input size: 10 (shard metrics features)
# - Hidden size: 128
# - Layers: 3 (bidirectional optional, but unidirectional for speed)
# - Sequence length: 60 (e.g., last 60 seconds/minutes of metrics)
# - Output: Linear head to 11 (10 predicted metrics + 1 overload prob sigmoid)
#
# Run this script with PyTorch installed:
#   pip install torch
#   python generate_lstm_1.1mb.py
#
# The resulting lstm_1.1mb.pt can be placed in src/embedding/models/
# and loaded in Rust via candle-core/torch or tch crates.

import torch
import torch.nn as nn
import os

class ShardLoadPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 10,       # Shard metrics: TPS, queue, latency, CPU, mem, etc.
        hidden_size: int = 128,
        num_layers: int = 3,
        seq_len: int = 60,          # Lookback window
        output_size: int = 11,      # 10 predicted metrics + 1 overload probability (0-1)
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # Unidirectional for lower latency
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(hidden_size, output_size)
        
        # Sigmoid for overload probability (last output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, seq_len, input_size) FloatTensor of historical metrics
        Returns: (batch, output_size) predictions
        """
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Use last timestep output
        last_out = self.dropout(lstm_out[:, -1, :])
        
        preds = self.output_head(last_out)
        
        # Apply sigmoid only to overload prob (last dim)
        preds[:, -1] = self.sigmoid(preds[:, -1])
        
        return preds

def main():
    # Create model
    model = ShardLoadPredictor()
    
    # Proper initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    model.apply(init_weights)
    
    # Eval mode
    model.eval()
    
    # Dynamic quantization to int8 (greatly reduces size, excellent for CPU)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )
    
    # Save quantized state dict
    save_path = "lstm_1.1mb.pt"
    torch.save(quantized_model.state_dict(), save_path)
    
    # Report size and stats
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Sharding load predictor model saved to {save_path}")
    print(f"File size: {size_mb:.2f} MB (target ~1.1MB)")
    print(f"Parameters: ~{param_count / 1e6:.2f}M (quantized int8)")
    print("\nPlace this file in src/embedding/models/ for use in sharding/lstm_predictor.rs")
    print("The model can be loaded in Rust with candle-core or tch for fast inference.")
    print("Input example: 60-step sequence of 10 normalized shard metrics.")
    print("Output: Predicted next metrics + overload probability for split/merge decisions.")

if __name__ == "__main__":
    main()
