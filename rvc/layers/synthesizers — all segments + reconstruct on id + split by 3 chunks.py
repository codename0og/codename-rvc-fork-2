import torch
from typing import Optional

from rvc.layers.algorithm.nsf import GeneratorNSF # GeneratorNSF
from rvc.layers.algorithm.generators import Generator #

from rvc.layers.algorithm.commons import slice_segments, rand_slice_segments
from rvc.layers.algorithm.residuals import ResidualCouplingBlock
from rvc.layers.algorithm.encoders import TextEncoder, PosteriorEncoder


class Synthesizer(torch.nn.Module):
    """
    Base Synthesizer model.

    Args:
        spec_channels (int): Number of channels in the spectrogram.
        segment_size (int): Size of the audio segment.
        inter_channels (int): Number of channels in the intermediate layers.
        hidden_channels (int): Number of channels in the hidden layers.
        filter_channels (int): Number of channels in the filter layers.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of layers in the encoder.
        kernel_size (int): Size of the convolution kernel.
        p_dropout (float): Dropout probability.
        resblock (str): Type of residual block.
        resblock_kernel_sizes (list): Kernel sizes for the residual blocks.
        resblock_dilation_sizes (list): Dilation sizes for the residual blocks.
        upsample_rates (list): Upsampling rates for the decoder.
        upsample_initial_channel (int): Number of channels in the initial upsampling layer.
        upsample_kernel_sizes (list): Kernel sizes for the upsampling layers.
        spk_embed_dim (int): Dimension of the speaker embedding.
        gin_channels (int): Number of channels in the global conditioning vector.
        sr (int): Sampling rate of the audio.
        use_f0 (bool): Whether to use F0 information.
        text_enc_hidden_dim (int): Hidden dimension for the text encoder.
        kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        use_f0,
        text_enc_hidden_dim=768,
        **kwargs
    ):
        super(Synthesizer, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.use_f0 = use_f0

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            text_enc_hidden_dim,
            f0=use_f0,
        )

        if use_f0:
            self.dec = GeneratorNSF(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                sr=sr,
                is_half=kwargs["is_half"],
            )
        else:
            self.dec = Generator(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
            )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = torch.nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        """Removes weight normalization from the model."""
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "WeightNorm"
            ):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        y: torch.Tensor = None,
        y_lengths: torch.Tensor = None,
        ds: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model with sequence restoration and regularized chunk shuffling.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            pitchf (torch.Tensor, optional): Fine-grained pitch sequence.
            y (torch.Tensor, optional): Target spectrogram.
            y_lengths (torch.Tensor, optional): Lengths of the target spectrograms.
            ds (torch.Tensor, optional): Speaker embedding. Defaults to None.
        """
        g = self.emb_g(ds).unsqueeze(-1)  # Embed speaker info (ds)
        
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        
        if y is not None:
            # Process target spectrogram y
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            
            # Slice the sequence into segments (all slices, not just one)
            z_slices, ids_slices = rand_slice_segments(z, y_lengths, self.segment_size)
            
            # Collect all the slices
            all_z_slices = []
            for i in range(z_slices.size(0)):  # Loop over all batches
                all_z_slices.append(z_slices[i])  # Collect slices for all batches
            
            # Stack all slices into a single tensor
            all_z_slices = torch.stack(all_z_slices, dim=0)
            
            # Shuffle the slices
            idx_shuffled = torch.randperm(all_z_slices.size(0), device=all_z_slices.device)
            all_z_slices = all_z_slices[idx_shuffled]
            ids_slices = ids_slices[idx_shuffled]  # Shuffle the corresponding ids_slices

            # Reorder the shuffled slices back to the original sequence using ids_slices
            _, reorder_indices = torch.sort(ids_slices)
            all_z_slices = all_z_slices[reorder_indices]
            
            # Regularization: Split the restored sequence into chunks and shuffle the chunks
            # Let's say we want to split the sequence into 3 chunks
            num_chunks = 3                                                            # adjust as you wish
            chunk_size = len(all_z_slices) // num_chunks  # Calculate chunk size

            # Ensure the chunks fit exactly
            chunked_slices = all_z_slices.split(chunk_size, dim=0)

            # If the chunking doesn't evenly divide, we pad the last chunk to fit
            if len(chunked_slices) > num_chunks:
                # If there's an extra chunk due to uneven split, pad it
                last_chunk = chunked_slices[-1]
                padded_chunk = torch.cat([last_chunk, torch.zeros_like(last_chunk)], dim=0)
                chunked_slices[-1] = padded_chunk

            # Shuffle the chunks
            shuffled_chunks = torch.randperm(len(chunked_slices), device=all_z_slices.device)

            # Reassemble the sequence with shuffled chunks
            shuffled_slices = torch.cat([chunked_slices[i] for i in shuffled_chunks], dim=0)
            
            # If using pitch, slice pitchf accordingly
            if self.use_f0 and pitchf is not None:
                pitchf_slices = slice_segments(pitchf, ids_slices, self.segment_size, 2)
                pitchf_slices = pitchf_slices[idx_shuffled]  # Shuffle pitchf slices to match z_slices
                
                # Reorder pitchf_slices based on the original order of ids_slices
                pitchf_slices = pitchf_slices[reorder_indices]

                # Decode using shuffled slices of z and pitchf
                o = self.dec(shuffled_slices, pitchf_slices, g=g)
            else:
                # Decode using shuffled slices of z
                o = self.dec(shuffled_slices, g=g)

            return o, ids_slices, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else:
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)


    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
    ):
        """
        Inference of the model.

        Args:
            phone (torch.Tensor): Phoneme sequence.
            phone_lengths (torch.Tensor): Lengths of the phoneme sequences.
            pitch (torch.Tensor, optional): Pitch sequence.
            nsff0 (torch.Tensor, optional): Fine-grained pitch sequence.
            sid (torch.Tensor): Speaker embedding.
            rate (torch.Tensor, optional): Rate for time-stretching. Defaults to None.
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if rate is not None:
            assert isinstance(rate, torch.Tensor)
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            if self.use_f0:
                nsff0 = nsff0[:, head:]
        if self.use_f0:
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, nsff0, g=g)
        else:
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)