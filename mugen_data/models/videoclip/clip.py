# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

from .modules import VideoEncoder, AudioEncoder, TextEncoder, ProjectionHead, Projection

class CLIPModel(nn.Module):
    def __init__(self, video_enc=False, audio_enc=False, text_enc=False, pretrained=False, trainable=False,
                 temperature=0.07, max_temperature=100.0, text_embedding=768):
        super().__init__()
        self.video_enc = video_enc
        self.audio_enc = audio_enc
        self.text_enc = text_enc

        if self.video_enc:
            self.visual_encoder = VideoEncoder(pretrained=pretrained, trainable=trainable)
            self.image_projection = Projection(self.visual_encoder.embedding_dim)
        if self.audio_enc:
            self.audial_encoder = AudioEncoder(pretrained=pretrained, trainable=trainable)
            self.audio_projection = Projection(self.audial_encoder.embedding_dim)
        if self.text_enc:
            self.text_encoder = TextEncoder(pretrained=pretrained, trainable=False)
            self.text_projection = Projection(text_embedding)

        if self.video_enc and self.audio_enc and self.text_enc:
            self.temperature_va = nn.Parameter(torch.tensor(temperature))
            self.temperature_vt = nn.Parameter(torch.tensor(temperature))
            self.temperature_at = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        self.max_temperature = nn.Parameter(torch.tensor(max_temperature), requires_grad=False)

    def get_video_embedding(self, batch):
        image_features = self.visual_encoder(batch["video"])
        image_embed = self.image_projection(image_features)
        image_embed = F.normalize(image_embed, dim=-1)
        return image_embed

    def get_audio_embedding(self, batch):
        audio_features = self.audial_encoder(batch["audio"])
        audio_embed = self.audio_projection(audio_features)
        audio_embed = F.normalize(audio_embed, dim=-1)
        return audio_embed

    def get_text_embedding(self, batch):
        text_features = self.text_encoder(batch['text'])
        # Getting Image and Text Embeddings (with same dimension)
        caption_embed = self.text_projection(text_features)
        caption_embed = F.normalize(caption_embed, dim=-1)
        return caption_embed

    def compute_similarities(self, batch, query_type, output_type):
        query_embed = getattr(self, f"get_{query_type}_embedding")(batch)
        output_embed = getattr(self, f"get_{output_type}_embedding")(batch)
        similarity = (query_embed @ output_embed.T) * torch.exp(torch.min(self.temperature, self.max_temperature))
        return similarity

    def forward(self, batch):
        if self.video_enc and not self.audio_enc and self.text_enc:
            similarity = self.compute_similarities(batch, "text", "video")
            loss = clip_loss(similarity)
            img_acc, cap_acc = metrics(similarity)
        elif self.video_enc and self.audio_enc and not self.text_enc:
            similarity = self.compute_similarities(batch, "audio", "video")
            loss = clip_loss(similarity)
            img_acc, cap_acc = metrics(similarity)
        elif self.audio_enc and not self.video_enc and self.text_enc:
            similarity = self.compute_similarities(batch, "text", "audio")
            loss = clip_loss(similarity)
            img_acc, cap_acc = metrics(similarity)
        elif self.video_enc and self.audio_enc and self.text_enc:
            vt_similarity = self.compute_similarities(batch, "text", "video")
            at_similarity = self.compute_similarities(batch, "text", "audio")
            va_similarity = self.compute_similarities(batch, "audio", "video")
            loss = clip_loss(vt_similarity)+clip_loss(at_similarity)+clip_loss(va_similarity)
            img_acc, cap_acc = metrics((vt_similarity+at_similarity+va_similarity)/3.0)
        else:
            raise NotImplementedError

        return loss, img_acc, cap_acc

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

def metrics(similarity):
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc