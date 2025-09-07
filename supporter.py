
import torch
import torch.nn as nn
from PIL import Image
import io
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10_000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 1024
NHEAD = 8
NUM_LAYERS = 3
DROPOUT = 0.1


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "<unk>": 3,
}
PAD_IDX = SPECIAL_TOKENS["<pad>"]
START_IDX = SPECIAL_TOKENS["<start>"]
END_IDX = SPECIAL_TOKENS["<end>"]
UNK_IDX = SPECIAL_TOKENS["<unk>"]



inference_transforms = v2.Compose([
    v2.Resize(IMAGE_SIZE),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]),
  ])

model_dict = torch.load("imageCaptioningModel_State.pth", map_location=device)


class EfficientNetEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, train_backbone=False):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features  # (B, 1280, H/32, W/32)
        for p in self.features.parameters():
            p.requires_grad = train_backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1280, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, 1280)
        x = self.proj(x)                           # (B, D)
        x = self.norm(x)
        x = x.unsqueeze(1)                         # treat as seq len 1 memory: (B, 1, D)
        return x

class CaptionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, nhead=NHEAD, ff_dim=FF_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, pad_idx=PAD_IDX):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_positions = nn.Embedding(SEQ_LENGTH, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, tgt_inp, memory):
        # tgt_inp: (B, L) token ids; memory: (B, M, D)
        B, L = tgt_inp.size()
        pos_ids = torch.arange(L, device=tgt_inp.device).unsqueeze(0).expand(B, L)
        x = self.embed_tokens(tgt_inp) + self.embed_positions(pos_ids)
        x = self.dropout(x)

        # causal mask (L, L) with True=mask
        causal_mask = torch.triu(torch.ones(L, L, device=tgt_inp.device) == 1, diagonal=1)

        # key padding mask for target (True for positions to mask)
        tgt_key_padding_mask = (tgt_inp == self.pad_idx)

        dec = self.decoder(
            x,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )
        logits = self.out(dec)
        return logits  # (B, L, V)

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EfficientNetEncoder(embed_dim=EMBED_DIM, train_backbone=False)
        self.decoder = CaptionTransformer(vocab_size=vocab_size)

    def forward(self, images, tgt_inp):
        memory = self.encoder(images)         # (B, 1, D)
        logits = self.decoder(tgt_inp, memory)  # (B, L, V)
        return logits


@torch.no_grad()
def generate_caption(model, image_bytes, itos, transforms=inference_transforms, max_len=SEQ_LENGTH):
    model.eval()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transforms(image)
    image = image.to(device).unsqueeze(0)
    feaures = model.encoder(image)  # (1, 1, D)
    
    tgt = torch.full((1, 1), START_IDX, dtype=torch.long, device=device)
    for _ in range(max_len-1):
        logits = model.decoder(tgt, feaures)  # (1, L, V)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        tgt = torch.cat([tgt, next_id], dim=1)
        if next_id.item() == END_IDX:
            break
        ids = tgt.squeeze(0).tolist()
        # drop <start>
        ids = ids[1:]
        # cut at <end>
        if END_IDX in ids:
            ids = ids[:ids.index(END_IDX)]
        
    tokens = [itos[str(i)] if 0 <= i < len(itos) else "<unk>" for i in ids]
    return " ".join(tokens)