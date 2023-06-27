# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import argparse
import itertools
import numpy as np
import os
from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from transformers import top_k_top_p_filtering
from models.gpt.attention import P2QAttentionStack, LayerNorm
from .utils import shift_dim, view_range
from tokenizers import Tokenizer

class MMGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.input_modality is not args.output_modality
        if 'video' in [args.input_modality, args.output_modality]:
            self.setup_video_vqvae()
            self.video_shape = self.video_vqvae.latent_shape
            self.video_seq_len = np.prod(self.video_shape)
            self.video_emb = nn.Linear(self.video_vqvae.embedding_dim, args.hidden_dim, bias=False)
            self.video_emb.weight.data.normal_(std=0.02)
            self.num_video_tokens = self.video_vqvae.n_codes
        if 'audio' in [args.input_modality, args.output_modality]:
            self.setup_audio_vqvae()
            self.num_audio_tokens = self.audio_vqvae.n_codes
            self.audio_shape = (int(self.audio_vqvae.z_shapes[0][0]),)
            audio_emb_dim = self.audio_vqvae.embedding_dim
            self.audio_seq_len = int(np.prod(self.audio_shape))
            self.audio_emb = nn.Linear(audio_emb_dim, args.hidden_dim, bias=False)
            self.audio_emb.weight.data.normal_(std=0.02)
        if 'text' in [args.input_modality, args.output_modality]:
            self.tokenizer = Tokenizer.from_file(args.tokenizer_file)
            self.pad_id = self.tokenizer.encode("[PAD]").ids[0]
            self.vocab_size = self.tokenizer.get_vocab_size()
            self.text_seq_len = args.text_seq_len
            self.num_text_tokens = self.vocab_size + args.text_seq_len
            self.text_emb = nn.Embedding(self.num_text_tokens, args.hidden_dim)

        self.attn_stack = P2QAttentionStack(
            args.hidden_dim, args.heads, args.layers, args.dropout, args.attn_dropout,
            video_shape=getattr(self, 'video_shape') if hasattr(self, 'video_shape') else None,
            audio_seq_len=getattr(self, 'audio_seq_len') if hasattr(self, 'audio_seq_len') else None,
            text_seq_len=getattr(self, 'text_seq_len') if hasattr(self, 'text_seq_len') else None,
            modality_p=self.args.input_modality, modality_q=self.args.output_modality
        )
        self.norm = LayerNorm(args.hidden_dim)

        self.num_input_tokens = getattr(self, f"num_{args.input_modality}_tokens")
        self.num_output_tokens = getattr(self, f"num_{args.output_modality}_tokens")

        total_tokens = self.num_input_tokens+self.num_output_tokens
        self.to_logit = nn.Linear(args.hidden_dim, total_tokens, bias=False)
        self.to_logit.weight.data.copy_(torch.zeros(total_tokens, args.hidden_dim))

        self.input_seq_len = getattr(self, f"{args.input_modality}_seq_len")
        self.output_seq_len = getattr(self, f"{args.output_modality}_seq_len")
        total_seq_len = self.input_seq_len+self.output_seq_len
        seq_range = rearrange(torch.arange(total_seq_len), 'n -> () n ()')
        logits_range = rearrange(torch.arange(total_tokens), 'd -> () () d')
        logits_mask = (
                ((seq_range >= self.input_seq_len) & (logits_range < self.num_input_tokens)) |
                ((seq_range < self.input_seq_len) & (logits_range >= self.num_input_tokens))
        )
        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.save_hyperparameters()

    def setup_video_vqvae(self):
        # Load VQ-VAE and set all parameters to no grad
        from models.video_vqvae.vqvae import VQVAE
        assert os.path.exists(self.args.video_vqvae)
        self.video_vqvae = VQVAE.load_from_checkpoint(self.args.video_vqvae)
        for p in self.video_vqvae.parameters():
            p.requires_grad = False
        self.video_vqvae.codebook._need_init = False
        self.video_vqvae.eval()

    def setup_audio_vqvae(self):
        # Load VQ-VAE and set all parameters to no grad
        from models.audio_vqvae.vqvae import VQVAE
        from jukebox.utils.torch_utils import freeze_model
        from models.audio_vqvae.hparams import setup_hparams
        hps = setup_hparams(self.args.audio_vqvae, {})
        self.audio_vqvae = VQVAE(hps)
        checkpoint = torch.load(hps.restore_vqvae, map_location=torch.device('cpu'))
        checkpoint['model'] = {k[7:] if k[:7] == 'module.' else k: v for k, v in checkpoint['model'].items()}
        self.audio_vqvae.load_state_dict(checkpoint['model'])
        if 'step' in checkpoint: self.audio_vqvae.step = checkpoint['step']
        self.audio_vqvae.eval()
        freeze_model(self.audio_vqvae)

    def text_to_tokens(self, x_text):
        all_tokens = [self.tokenizer.encode(t.strip().lower()+" [SEP]") for t in x_text]
        x_text = [t.ids[:self.text_seq_len] for t in all_tokens]
        context_len = self.text_seq_len
        for i, t in enumerate(x_text):
            t += [self.pad_id] * (context_len - len(t))
            x_text[i] = t
        return torch.Tensor(x_text).type(torch.int64)


    def forward(self, x_input, x_output, targets, decode_step=None, decode_idx=None):
        h_input = getattr(self, f"{self.args.input_modality}_emb")(x_input)
        h_output = getattr(self, f"{self.args.output_modality}_emb")(x_output)

        h = self.attn_stack(h_input, h_output, decode_step, decode_idx)
        h = self.norm(h)


        logits = self.to_logit(h)
        seq_len = self.input_seq_len + self.output_seq_len
        logits_mask = torch.cat((self.logits_mask[:, :self.input_seq_len],
                                 self.logits_mask[:, self.input_seq_len:seq_len]), dim=1)
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)
        logits = shift_dim(logits, -1, 1)
        loss_input = F.cross_entropy(logits[:, :, :self.input_seq_len], targets[:, :self.input_seq_len])
        loss_output = F.cross_entropy(logits[:, :, self.input_seq_len:], targets[:, self.input_seq_len:])
        in_weight = getattr(self.args, f"loss_{self.args.input_modality}_weight")
        out_weight = getattr(self.args, f"loss_{self.args.output_modality}_weight")
        loss = (in_weight * loss_input + out_weight * loss_output) / (in_weight + out_weight)

        return loss, loss_input, loss_output

    def encode(self, x, modality, device):
        if modality == 'video':
            with torch.no_grad():
                v_tok = self.video_vqvae.encode(x).flatten(start_dim=1, end_dim=-1)
                v_emb = self.video_vqvae.codebook.dictionary_lookup(v_tok)
                return v_emb, v_tok
        elif modality == 'audio':
            with torch.no_grad():
                a_tok, a_emb = self.audio_vqvae.encode(x, include_embeddings=True)
                return shift_dim(a_emb, 1, -1), a_tok
        elif modality == 'text':
            assert self.tokenizer is not None
            t_tok = self.text_to_tokens(x).to(device)
            assert t_tok.shape[-1] == self.text_seq_len, f'text input shape mismatch, input {x.shape[-1]}, expected ({self.text_seq_len})'
            # make sure padding in text tokens get unique padding token id
            text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
            t_tok = torch.where(t_tok == self.pad_id, text_range, t_tok)
            return t_tok, t_tok

    def lookup(self, sample_idxs):
        if self.args.output_modality == 'video':
            embeddings_slice = self.video_vqvae.codebook.dictionary_lookup(sample_idxs)
        elif self.args.output_modality == 'audio':
            embeddings_slice = self.audio_vqvae.bottleneck.decode([sample_idxs])[0].permute(0, 2, 1)
        elif self.args.output_modality == 'text':
            embeddings_slice = sample_idxs
        return embeddings_slice

    def decode(self, x, modality):
        import ipdb;ipdb.set_trace()
        if modality == 'video':
            samples = self.video_vqvae.decode(view_range(x[:, self.input_seq_len:] - self.num_input_tokens, 1, None, self.video_shape))
            samples = self.video_vqvae.postprocess(samples)
            return samples
        elif modality == 'audio':
            samples = self.audio_vqvae.decode(view_range(x[:, self.input_seq_len:] - self.num_input_tokens, 1, None, self.audio_shape))
            samples = torch.clamp(samples, -0.5, 0.5)
            return samples  # BCTHW in [0, 1]
        elif modality == 'text':
            sentences = [self.tokenizer.decode([t for t in sample.tolist() if t <= self.vocab_size and t > 0]) for sample in (x[:,self.input_seq_len:] - self.num_input_tokens)]
            return sentences

    def shared_step(self, batch, batch_idx):
        if self.args.video_vqvae is not None:
            self.video_vqvae.eval() # this is required even though we set vqvae to eval after setup.
        if self.args.audio_vqvae is not None:
            self.audio_vqvae.eval() # this is required even though we set vqvae to eval after setup.
        sources, targets = {}, {}
        device = batch['video'].device if 'video' in batch else batch['audio'].device
        sources[self.args.input_modality], targets[self.args.input_modality] = self.encode(batch[self.args.input_modality], self.args.input_modality, device)
        sources[self.args.output_modality], targets[self.args.output_modality] = self.encode(batch[self.args.output_modality], self.args.output_modality, device)

        targets = torch.cat((targets[self.args.input_modality], targets[self.args.output_modality] + self.num_input_tokens), 1)
        loss, text_loss, img_loss = self(sources[self.args.input_modality], sources[self.args.output_modality], targets)
        return loss, text_loss, img_loss

    def training_step(self, batch, batch_idx):
        import ipdb;ipdb.set_trace()
        loss, input_loss, output_loss = self.shared_step(batch, batch_idx)
        self.log(f"train/{self.args.input_modality}_loss", input_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"train/{self.args.output_modality}_loss", output_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        import ipdb;ipdb.set_trace()
        loss, text_loss, img_loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True)
        self.log(f"val/{self.args.input_modality}_loss", text_loss, prog_bar=True)
        self.log(f"val/{self.args.output_modality}_loss", img_loss, prog_bar=True)

    def sample(self, n, batch=None, top_k=None, top_p=None):
        x_in = batch[self.args.input_modality]
        device = 'cuda'
        x_emb, x_tok = self.encode(x_in, self.args.input_modality, device)
        samples = torch.cat((x_tok, self.num_input_tokens * torch.ones((n,) + (self.output_seq_len,)).long().to(device)), 1)
        idxs = list(itertools.product(*[range(s) for s in (self.input_seq_len + self.output_seq_len,)]))
        with torch.no_grad():
            prev_idx = None
            input_embeddings = getattr(self, f"{self.args.input_modality}_emb")(x_emb)
            for decode_step, decode_idx in enumerate(idxs):
                batch_idx_slice = (
                    slice(None, None), *[slice(decode_step, decode_step + 1) for decode_step in decode_idx])
                batch_idx = (slice(None, None), *decode_idx)

                if decode_step <= self.input_seq_len:
                    if prev_idx is None:
                        h_in = input_embeddings[batch_idx_slice]
                    else:
                        h_in = input_embeddings[prev_idx]
                    h = self.attn_stack(h_in, None, decode_step, decode_idx)
                else:
                    output_sample = samples[:, self.input_seq_len:] - self.num_input_tokens
                    output_prev_idx = (slice(None, None), slice(prev_idx[1].start - self.input_seq_len,
                                                               prev_idx[1].stop - self.input_seq_len))
                    embeddings_slice = self.lookup(output_sample[output_prev_idx])
                    h_out = getattr(self, f"{self.args.output_modality}_emb")(embeddings_slice)
                    h = self.attn_stack(None, h_out, decode_step, decode_idx)
                if decode_step >= self.input_seq_len:
                    h = self.norm(h)
                    logits = self.to_logit(h)
                    logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                    max_neg_value = -torch.finfo(logits.dtype).max
                    output_mask = torch.zeros_like(logits).bool()
                    output_mask[:, :self.num_input_tokens] = True
                    logits.masked_fill_(output_mask, max_neg_value)
                    if top_k is not None or top_p is not None:
                        logits_output = logits[:, self.num_input_tokens:]
                        logits_output = top_k_top_p_filtering(logits_output, top_k=top_k, top_p=top_p)
                        logits[:, self.num_input_tokens:] = logits_output
                    probs = F.softmax(logits, dim=-1)
                    samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)
                prev_idx = batch_idx_slice
            samples = self.decode(samples, self.args.output_modality)
        return samples# BCTHW in [0, 1]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_modality", type=str, default=None, choices=['video', 'audio', 'text'])
        parser.add_argument("--output_modality", type=str, default=None, choices=['video', 'audio', 'text'])
        parser.add_argument('--hidden_dim', type=int, default=768)
        parser.add_argument('--heads', type=int, default=8)
        parser.add_argument('--layers', type=int, default=12)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--attn_dropout', type=float, default=0.3)
        parser.add_argument('--loss_video_weight', type=float, default=1)
        parser.add_argument('--loss_audio_weight', type=float, default=1)
        parser.add_argument('--loss_text_weight', type=float, default=1)
        parser.add_argument("--lr", type=float, default=0.0003)

        parser.add_argument('--video_vqvae', type=str, default=None, help='path to video vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--audio_vqvae', type=str, default=None, help='audio vqvae model name')
        parser.add_argument("--tokenizer_file", type=str, default="datasets/coinrun/tokenizers/tokenizer-coinrun_1024.json")
        parser.add_argument('--text_seq_len', type=int, default=128)
        return parser

