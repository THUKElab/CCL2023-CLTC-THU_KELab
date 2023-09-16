import torch
from typing import Optional
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerEncoder, TransformerModel, base_architecture

from utils import get_logger

LOGGER = get_logger(__name__)


class GECTransformerEncoder(TransformerEncoder):
    """ 相比传统的 Transformer Encoder 的改动：
        1）增加 src_dropout
    """

    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        super().__init__(args, dictionary, embed_tokens, return_fc)
        self.src_dropout = args.source_word_dropout
        LOGGER.info(f"Use src_dropout: {self.src_dropout}")

    def SRC_dropout(self, embedding_tokens, drop_prob):
        if drop_prob == 0:
            return embedding_tokens
        keep_prob = 1 - drop_prob
        mask = (torch.randn(embedding_tokens.size()[:-1]) < keep_prob).unsqueeze(-1)
        embedding_tokens *= mask.eq(1).to(embedding_tokens.device)
        return embedding_tokens * (1 / keep_prob)

    def forward_embedding(
            self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        if self.training:
            token_embedding = self.SRC_dropout(token_embedding, self.src_dropout)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


@register_model("gec_transformer")
class GECTransformer(TransformerModel):
    """ 相比传统的 Transformer Encoder 的改动：
        1）使用 GECTransformerEncoder
        2) 提供 set_beam_size 方法支持 gec_dev 实时评估 F0.5
    """

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument(
            '--source-word-dropout',
            type=float, metavar='D', default=0.2,
            help='dropout probability for source word dropout',
        )

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return GECTransformerEncoder(cfg, src_dict, embed_tokens)

    def set_beam_size(self, beam):
        """Set beam size for efficient beamable enc-dec attention."""
        beamable = False
        for layer in self.decoder.layers:
            if layer.encoder_attn is not None:
                if hasattr(layer.encoder_attn, "set_beam_size"):
                    layer.encoder_attn.set_beam_size(beam)
                    beamable = True
        if beamable:
            self.encoder.reorder_encoder_out = self.encoder._reorder_encoder_out


@register_model_architecture("gec_transformer", "gec_transformer")
def gec_transformer_base_architecture(args):
    base_architecture(args)
    args.source_word_dropout = getattr(args, "source_word_dropout", 0.2)
