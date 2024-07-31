import torch
from torch import nn, Tensor
from zeta import (
    FeedForward,
    MultiQueryAttention,
    MambaBlock,
    OutputHead,
)
from typing import Any


class AttentionBlock(nn.Module):
    """
    AttentionBlock class implements a block with attention mechanism.

    Args:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float): The dropout rate.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension of the block.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        dropout (float): The dropout rate.
        attn (MultiQueryAttention): The multi-query attention module.
        ffn (FeedForward): The feedforward module.

    Methods:
        forward(x, mask=None): Performs a forward pass through the block.

    """

    def __init__(
        self,
        dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        # Attention
        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        # Feedforward
        self.ffn = FeedForward(
            dim,
            dim,
            4,
            swish=True,
            dropout=dropout,
        )

    def forward(self, x: Tensor, mask: Tensor = None):
        """
        Performs a forward pass through the AttentionBlock.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after passing through the AttentionBlock.

        """
        residual = x
        print(x.shape)
        x, _, _ = self.attn(x, mask=mask)
        print(x.shape)

        out = x + residual
        print(out.shape)

        # FFN
        residual_two = out
        print(residual_two.shape)
        feeded = self.ffn(out)
        print(feeded.shape)

        # + residual_two

        return feeded + residual_two


# x = torch.randn(1, 10, 512)
# mask = torch.zeros(1, 10)
# block = AttentionBlock(dim=512)
# output = block(x)
# print(output.shape)


class LSTMBlock(nn.Module):
    """
    LSTMBlock is a module that combines LSTM and FeedForward layers.

    Args:
        dim (int): The input dimension.
        hidden_size (int): The hidden size of the LSTM layer.
        num_layers (int): The number of LSTM layers.
        dropout (float): The dropout probability.
        ff_hidden_size (int): The hidden size of the FeedForward layer.

    Attributes:
        dim (int): The input dimension.
        hidden_size (int): The hidden size of the LSTM layer.
        num_layers (int): The number of LSTM layers.
        dropout (float): The dropout probability.
        ff_hidden_size (int): The hidden size of the FeedForward layer.
        lstm (nn.LSTM): The LSTM layer.
        ffn (FeedForward): The FeedForward layer.

    Methods:
        forward(x, mask=None): Performs the forward pass of the LSTMBlock.

    """

    def __init__(
        self,
        dim: int = None,
        hidden_size: int = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            dim,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            *args,
            **kwargs,
        )

        # FFN
        self.ffn = FeedForward(
            dim,
            dim,
            4,
            swish=True,
            dropout=dropout,
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Performs the forward pass of the LSTMBlock.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.

        """
        residual = x
        lstm_out, _ = self.lstm(x)

        lstm_out += residual

        # Feedforwarded
        second_residual = lstm_out

        # FFN
        feeded = self.ffn(lstm_out) + second_residual
        droped = nn.Dropout(self.dropout)(feeded)

        return nn.LayerNorm(self.dim)(droped + residual)


# x = torch.randn(1, 10, 512)
# mask = torch.zeros(1, 10)
# block = LSTMBlock(dim=512, hidden_size=512)
# output = block(x)
# print(output)


class VortexFusion(nn.Module):
    def __init__(
        self,
        dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        lstm_num_layers: int = 1,
        lstm_hidden_size: int = 512,
        attention_block_layers: int = 4,
        lstm_block_layers: int = 4,
        d_state: int = 512,
        d_conv: int = 512,
        mamba_block_layers: int = 4,
        vocab_size: int = 10000,
        ffn_bride: bool = True,
        post_embed_norm: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = vocab_size
        self.ffn_bride = ffn_bride
        self.post_embed_norm = post_embed_norm

        # Attention Block Layers
        self.attn_block_layers = nn.ModuleList(
            [
                AttentionBlock(
                    dim, heads, dim_head, dropout, *args, **kwargs
                )
                for _ in range(attention_block_layers)
            ]
        )

        # LSTM Block Layers
        self.lstm_block_layers = nn.ModuleList(
            [
                LSTMBlock(
                    dim,
                    lstm_hidden_size,
                    lstm_num_layers,
                    dropout,
                    *args,
                    **kwargs,
                )
                for _ in range(lstm_block_layers)
            ]
        )

        # Mamba
        self.mamba_block_layers = nn.ModuleList(
            [
                MambaBlock(
                    dim,
                    depth=1,
                    d_state=d_state,
                    d_conv=d_conv,
                )
                for _ in range(mamba_block_layers)
            ]
        )

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, dim)

        # FFN
        self.ffn = FeedForward(
            dim,
            dim,
            2,
            swish=True,
            dropout=dropout,
        )

        # Maybe have a bridge between models like a linear projec or a feedforward layer

    def forward(self, x: Any, mask: Tensor = None) -> Tensor:
        x = self.embed(x)

        if self.post_embed_norm is True:
            x = nn.LayerNorm(self.dim)(x)

        # Now we need to run a loop where mamba -> transformer -> lstm -> mamba
        for attn_block, lstm_block, mamba_block in zip(
            self.attn_block_layers,
            self.lstm_block_layers,
            self.mamba_block_layers,
        ):
            residual = x
            x = mamba_block(x) + residual
            x = self.ffn(x)

            # 2nd residual
            residual_two = x
            x = attn_block(x) + residual_two
            x = self.ffn(x)

            # 3rd residual
            residual_three = x
            x = lstm_block(x) + residual_three

        return OutputHead(self.dim, 1, self.vocab_size)(self.ffn(x))


# x = torch.randint(0, 10000, (1, 10))
# model = VortexFusion(dim=512)
# output = model(x)
# print(output.shape)
