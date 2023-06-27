# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

from jukebox.vqvae.vqvae import VQVAE as Jukebox_VQVAE
from jukebox.utils.audio_utils import audio_preprocess

class VQVAE(Jukebox_VQVAE):
    def __init__(self, hps):
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv,
                            dilation_growth_rate=hps.dilation_growth_rate,
                            dilation_cycle=hps.dilation_cycle,
                            reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

        super().__init__(
            input_shape = (hps.sample_length, 1), levels = hps.levels, downs_t = hps.downs_t, strides_t = hps.strides_t,
            emb_width = hps.emb_width, l_bins = hps.l_bins, mu = hps.l_mu, commit = hps.commit, spectral = hps.spectral,
            multispectral = hps.multispectral, multipliers = hps.hvqvae_multipliers, use_bottleneck = hps.use_bottleneck,
            **block_kwargs)
        self.n_codes = hps.l_bins
        self.embedding_dim = hps.emb_width
        self.hps = hps

    def decode(self, zs, start_level=0, end_level=None):
        return self._decode([zs], start_level, end_level)

    def encode(self, x, start_level=0, end_level=None, include_embeddings=False):
        x = audio_preprocess(x, self.hps)
        zs = self._encode(x, start_level, end_level)
        if include_embeddings:
            embeddings = self.bottleneck.decode(zs)
            return zs[start_level], embeddings[start_level]
        return zs[start_level]
