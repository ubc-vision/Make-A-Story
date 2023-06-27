# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import torch
from torch import nn
from transformers import DistilBertModel, DistilBertConfig
from .s3d import S3D
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize
from einops import rearrange
from transformers import DistilBertTokenizer
import numpy as np
from scipy import signal

class Projection(nn.Module):
    def __init__(self, d_in, d_out=256, p=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

from .resnet import resnet18
class AudioEncoder(nn.Module):
    """
    Encode audios to a fixed size vector
    """

    def __init__(self, pretrained, trainable):
        super(AudioEncoder, self).__init__()
        self.model = resnet18(num_classes=309)# check later
        self.embedding_dim = self.model.fc.in_features
        if pretrained:
            print("Loading pretrained ResNet18 from H.pth.tar")
            weight_dict = torch.load("checkpoints/pretrained/H.pth.tar")['model_state_dict']
            model_dict = self.model.state_dict()
            for name, param in weight_dict.items():
                if 'audnet' in name:
                    name = '.'.join(name.split('.')[1:])
                model_dict[name].copy_(param)
        self.model.fc = nn.Identity()
        for p in self.model.parameters():
            p.requires_grad = trainable

    def preprocess(self, x):
        sr = 16000
        device = x.device
        resamples = F.interpolate(torch.mean(x, -1)[:,None], int(sr*3.2)).cpu().numpy()
        resamples = resamples + 0.5
        resamples = np.tile(resamples, 10)
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        spectrograms = torch.from_numpy(signal.spectrogram(resamples, sr, nperseg=512, noverlap=353)[-1])
        spectrograms = torch.log(spectrograms + 1e-7)
        mean, std = torch.mean(spectrograms.flatten(1), dim=-1), torch.std(spectrograms.flatten(1), dim=-1)
        spectrograms = ((spectrograms - mean[:,None,None,None]) / (std[:,None,None,None] + 1e-9)).float().to(device)
        return spectrograms

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)

class VideoEncoder(nn.Module):
    """
    Encode videos to a fixed size vector
    """

    def __init__(self, pretrained, trainable):
        super().__init__()

        self.model = S3D(400)
        self.embedding_dim = list(self.model.fc.children())[0].in_channels
        if pretrained:
            print("Loading pretrained S3D from S3D_kinetics400.pt")
            weight_dict = torch.load('checkpoints/pretrained/S3D_kinetics400.pt')
            model_dict = self.model.state_dict()
            for name, param in weight_dict.items():
                if 'module' in name:
                    name = '.'.join(name.split('.')[1:])
                model_dict[name].copy_(param)
            
        self.model.fc = nn.Identity()
        for p in self.model.parameters():
            p.requires_grad = trainable
        
    def preprocess(self, x):
        B, T, H, W, C = x.shape
        if T != 32:
            x = F.interpolate(rearrange(x, "b t h w c -> b c t h w"), size=[32, H, W])
            x = rearrange(x, "b c t h w -> b t h w c")
        assert C == 3
        x = rearrange(x, "b t h w c -> (b t) c h w")
        x = resize(x, (224, 224)) if H != 224 and W != 224 else x
        # this is a rgb video, just normalize
        x = x.float() / 255.
        # convert to BCTHW
        x = normalize(x, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989))
        x = rearrange(x, "(b t) c h w -> b c t h w", b = B)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=True, max_length=200):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, raw_text):
        batch_encoding = self.tokenizer(raw_text, padding=True, truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(batch_encoding['input_ids']).cuda()
        attention_mask = torch.tensor(batch_encoding['attention_mask']).cuda()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

