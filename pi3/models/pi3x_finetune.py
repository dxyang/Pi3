import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from .pi3x import Pi3X
from .lora import apply_lora


def freeze(*modules):
    for m in modules:
        if isinstance(m, nn.Parameter):
            m.requires_grad = False
        else:
            for p in m.parameters():
                p.requires_grad = False


class Pi3XFinetune(Pi3X):
    """Pi3X wired for fine-tuning through the stock Pi3 trainer.

    Subclasses rather than wraps Pi3X so `model.encoder` stays reachable --
    Pi3Trainer.build_optimizer splits parameter groups on that attribute.

    Vision-only arm: forward() is called with images alone, which drives every
    conditioning probability in encode() to zero, so the depth/ray/pose branches
    are skipped outright. Those branches and the metric head must nonetheless be
    frozen explicitly: params that stay trainable but receive no gradient make DDP
    raise unless find_unused_parameters is on (it is off in the stock config).
    Freezing them also keeps the trainable-parameter count meaningful.

    The metric head is unsupervised until the metric-scale term from
    pi3x_loss.py:165-173 is ported, so it is frozen by default rather than left
    to drift while everything around it moves.
    """

    def __init__(
        self,
        pretrained='yyfz233/Pi3X',
        ckpt=None,
        freeze_encoder=True,
        freeze_multimodal=True,
        freeze_metric=True,
        freeze_decoder=False,        # the 453.5M shared trunk -- 74% of trainable params
        freeze_task_decoders=False,  # point_decoder + camera_decoder (131.7M)
        lora_r=0,                    # >0: freeze the trunk and adapt it with LoRA instead
        lora_alpha=32,
        train_conf=False,
    ):
        super().__init__()

        if pretrained:
            path = hf_hub_download(repo_id=pretrained, filename='model.safetensors')
            res = self.load_state_dict(load_file(path), strict=True)
            print(f'[Pi3XFinetune] loaded {pretrained}: {res}')

        self._pending_ckpt = ckpt

        self.train_conf = train_conf

        if freeze_encoder:
            freeze(self.encoder)
        if freeze_multimodal:
            freeze(self.depth_encoder, self.ray_embed, self.pose_inject_blk, self.depth_emb)
        if freeze_metric:
            freeze(self.metric_decoder, self.metric_head, self.metric_token)
        if not train_conf:
            # Pi3Loss ignores pred['conf'] unless train_conf is set, so these would
            # be trainable but ungradiented -- the same DDP hazard as the metric head.
            freeze(self.conf_decoder, self.conf_head)
        if freeze_task_decoders:
            freeze(self.point_decoder, self.camera_decoder)

        # Capacity control. The trunk is 453.5M of 613.8M trainable, against one
        # dive's worth of highly correlated frames -- full fine-tuning overfits by
        # epoch 2 while local_pts_loss never improves at all. LoRA lets the trunk
        # still adapt at ~2% of the parameter count; freeze_decoder removes it
        # from training entirely.
        if lora_r > 0:
            freeze(self.decoder)
            n = apply_lora(self.decoder, r=lora_r, alpha=lora_alpha)
            print(f'[Pi3XFinetune] LoRA r={lora_r} alpha={lora_alpha} on {n} decoder layers')
        elif freeze_decoder:
            freeze(self.decoder)

        # Load the fine-tuned checkpoint AFTER any LoRA wrapping, so the key
        # layout matches. A silent strict=False load here would leave the trunk
        # at pretrained values while the heads expect an adapted one.
        if ckpt is not None:
            state = torch.load(ckpt, weights_only=False, map_location='cpu')
            state = state.get('model', state)
            res = self.load_state_dict(state, strict=False)
            n_missing, n_unexpected = len(res.missing_keys), len(res.unexpected_keys)
            print(f'[Pi3XFinetune] resume from {ckpt}: '
                  f'{n_missing} missing, {n_unexpected} unexpected')
            if n_missing or n_unexpected:
                raise RuntimeError(
                    f'Checkpoint/architecture mismatch ({n_missing} missing, '
                    f'{n_unexpected} unexpected). e.g. missing={res.missing_keys[:2]} '
                    f'unexpected={res.unexpected_keys[:2]}. Check lora_r and freeze flags.')

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'[Pi3XFinetune] trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params')

    def forward(self, imgs, **kwargs):
        out = super().forward(imgs, **kwargs)
        # Pi3Loss reads the world-frame pointmap as 'global_points'; Pi3X calls it
        # 'points'. Same tensor -- expose both so the global term is supervised.
        out['global_points'] = out['points']
        return out
