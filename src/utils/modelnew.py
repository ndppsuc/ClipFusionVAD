from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head,dropout=0.1)
        self.weight = nn.Parameter(torch.ones(n_head))
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
            ("dropout", nn.Dropout(0.1))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
#创新1：引入了dynamic attention
    #创新2：引入了dropout
    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        # 通过动态调整权重影响注意力机制
        weight = F.softmax(self.weight, dim=0)
        attn_output, _ = self.attn(x, x, x, attn_mask=None, key_padding_mask=padding_mask)
        return weight.view(-1, 1, 1) * attn_output  # 动态加权

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        # 空间注意力保持多尺度卷积
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=3, padding=1),
                nn.BatchNorm2d(int(in_channels / rate)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=5, padding=2),
                nn.BatchNorm2d(int(in_channels / rate)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(in_channels)
            )
        ])

    def forward(self, x):
        # 通道注意力
        b, c, h, w = x.shape
        x_permute = x.view(b, c, -1)
        x_att_permute = self.channel_attention(x_permute).view(b, c, h, w)
        x_channel_att = x_att_permute.sigmoid()

        # 结合通道注意力和空间注意力
        x = x * x_channel_att  # 先进行通道注意力加权

        # 然后进行空间注意力处理
        x_spatial_att = [spatial_att(x) for spatial_att in self.spatial_attention]
        x_spatial_att = sum(x_spatial_att)
        x_spatial_att = x_spatial_att.sigmoid()

        out = x * x_spatial_att  # 最终融合
        return out


class BilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(BilinearPooling, self).__init__()
        # 线性层将模态特征映射到相同的维度
        self.fc1 = nn.Linear(input_dim1, output_dim)
        self.fc2 = nn.Linear(input_dim2, output_dim)

    def forward(self, x1, x2):
        # 线性变换
        x1_proj = self.fc1(x1)
        x2_proj = self.fc2(x2)

        # 元素乘积（双线性融合）
        bilinear_product = torch.bmm(x1_proj.unsqueeze(2), x2_proj.unsqueeze(1))
        return bilinear_product.squeeze()


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 device):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )
        self.gam_attention = GAM_Attention(in_channels=visual_width)

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.bilinear_pooling = BilinearPooling(visual_width, embed_dim, output_dim=visual_width)
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.head_video = nn.Linear(embed_dim, embed_dim)
        self.u_head_video = nn.Linear(embed_dim, embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat(
            [len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def uda(self, video_feature, text_feature, train_flag):
        v_fea = self.head_video(video_feature)
        v_fea_u = self.u_head_video(video_feature)

        v_fea = v_fea / v_fea.norm(dim=-1, keepdim=True)
        v_fea_u = v_fea_u / v_fea_u.norm(dim=-1, keepdim=True)
        t_fea = text_feature / text_feature.norm(dim=-1, keepdim=True)

        if train_flag:
            v_fea_u_nograd = self.u_head_video(video_feature.detach())
            t_fea_nograd = t_fea.detach()
            return video_feature, v_fea, v_fea_u, t_fea, v_fea_u_nograd, t_fea_nograd
        else:
            return video_feature, v_fea, v_fea_u, t_fea, None, None

    def forward(self, visual, padding_mask, text, lengths, train_flag=False):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        visual_features = self.gam_attention(visual_features.permute(0, 2, 1)).permute(0, 2, 1)

        video_feature, v_fea, v_fea_u, t_fea, v_fea_u_nograd, t_fea_nograd = self.uda(visual_features,
                                                                                      self.encode_textprompt(text),
                                                                                      train_flag)
        logits1 = self.classifier(visual_features + self.mlp2(v_fea + v_fea_u))

        text_features_ori = self.encode_textprompt(text)

        text_features = t_fea
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ v_fea_u
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], t_fea.shape[0], visual_attn.shape[2])
        text_features = t_fea.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        # text_features = text_features + self.mlp1(text_features)
        text_features = self.bilinear_pooling(visual_features, text_features)
        visual_features_norm = v_fea / v_fea.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2
