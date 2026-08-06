"""PVNet 네트워크: 픽셀별 세그멘테이션 + 키포인트 방향 벡터장 예측.

Peng et al., CVPR 2019, Sec 3.1(방식) 및 Sec 4(구현) 구조를 따르되,
RTMDet-Ins가 이미 인스턴스를 분리해 크롭을 넘겨주는 지금 파이프라인에
맞춰 단순화했다:

    - 원 논문은 이미지 전체에서 멀티클래스+멀티인스턴스를 한 번에 처리하지만
      (Sec 3.1 "Multiple instances", 인스턴스 중심을 투표로 찾아 분리),
      여기서는 rotation_head_model.py의 crop_and_preprocess()와 동일하게
      "인스턴스 하나 = 크롭 하나"로 입력받는다. 따라서 세그멘테이션은
      배경/전경 2클래스면 충분하고, 인스턴스 분리 로직은 필요 없다
      (그 역할은 이미 RTMDet-Ins가 한다).
    - backbone은 RotationHeadNet과 동일하게 resnet18로 통일해 두 헤드를
      나란히 두고 비교/교체하기 쉽게 했다.

출력 해상도는 입력과 동일(H, W) - 벡터장과 세그멘테이션 둘 다 픽셀 단위
예측이어야 voting.py의 RANSAC 투표가 의미를 가진다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DEFAULT_BACKBONE = "resnet18"
DEFAULT_PRETRAINED = True
DEFAULT_NUM_KEYPOINTS = 9  # keypoints.py 기본값(표면 8 + 센트로이드 1)과 맞춤


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PVNetHead(nn.Module):
    """단일 인스턴스 크롭 -> (세그멘테이션 로짓, 키포인트 방향 벡터장).

    입력: (B, 3, H, W) float32, [0, 1] 정규화. rotation_head_model.py의
        crop_and_preprocess() 출력을 그대로 재사용할 수 있다(mask_background=True
        권장 - 배경 픽셀이 지워져 있으면 세그멘테이션 학습이 쉬워짐).

    출력:
        seg_logits: (B, 2, H, W) - 배경(채널 0)/전경(채널 1) 로짓.
        vertex: (B, K*2, H, W) - 픽셀마다 키포인트 k를 향하는 방향벡터의
            (dx, dy). 채널 순서는 [k0_dx, k0_dy, k1_dx, k1_dy, ...] 이며
            keypoints.py가 반환한 키포인트 순서와 반드시 일치해야 한다.
            추론 시 이 벡터가 unit일 필요는 없다(voting.py에서 정규화) -
            논문 Sec 4.1에서도 동일하게 명시.
    """

    def __init__(
        self,
        num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = DEFAULT_PRETRAINED,
    ):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"지원하지 않는 backbone: {backbone} (지금은 resnet18만 지원)")
        self.num_keypoints = num_keypoints

        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)

        # encoder: 중간 스테이지 출력을 각각 붙잡아 표준 U-Net 스타일 디코더의
        # 스킵 커넥션으로 사용한다. 논문 원본은 stride 8 이후 풀링을 없애고
        # dilated conv로 대체하지만, 여기서는 스킵 커넥션 업샘플링 쪽이
        # 구현/디버깅이 쉬워 이 방식을 택했다 (결과 해상도는 동일하게 HxW).
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # stride 2,  64ch
        self.pool = net.maxpool                                  # stride 4
        self.layer1 = net.layer1                                 # stride 4,  64ch
        self.layer2 = net.layer2                                 # stride 8,  128ch
        self.layer3 = net.layer3                                 # stride 16, 256ch
        self.layer4 = net.layer4                                 # stride 32, 512ch

        # 디코더: stride 32 -> 16 -> 8 -> 4, 매 단계 스킵 커넥션 concat 후 conv.
        self.up4 = _ConvBnRelu(512 + 256, 256)
        self.up3 = _ConvBnRelu(256 + 128, 128)
        self.up2 = _ConvBnRelu(128 + 64, 64)
        self.up1 = _ConvBnRelu(64 + 64, 64)

        out_ch = 2 + num_keypoints * 2
        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]

        s2 = self.stem(x)       # stride 2
        s4a = self.pool(s2)
        s4 = self.layer1(s4a)   # stride 4
        s8 = self.layer2(s4)    # stride 8
        s16 = self.layer3(s8)   # stride 16
        s32 = self.layer4(s16)  # stride 32

        d16 = self._up_concat(s32, s16, self.up4)
        d8 = self._up_concat(d16, s8, self.up3)
        d4 = self._up_concat(d8, s4, self.up2)
        d2 = self._up_concat(d4, s2, self.up1)

        out = F.interpolate(d2, size=(h, w), mode="bilinear", align_corners=False)
        out = self.out_conv(out)

        seg_logits = out[:, :2]
        vertex = out[:, 2:]
        return seg_logits, vertex

    @staticmethod
    def _up_concat(x_low: torch.Tensor, x_skip: torch.Tensor, block: nn.Module) -> torch.Tensor:
        x_up = F.interpolate(x_low, size=x_skip.shape[-2:], mode="bilinear", align_corners=False)
        return block(torch.cat([x_up, x_skip], dim=1))


# =============================================================================
# 학습 loss (논문 Sec 4.1)
# =============================================================================
def vertex_smooth_l1_loss(
    vertex_pred: torch.Tensor,
    vertex_gt: torch.Tensor,
    seg_gt: torch.Tensor,
) -> torch.Tensor:
    """벡터장 smooth L1 loss, 전경 픽셀만 (논문 Eq.6).

    Args:
        vertex_pred, vertex_gt: (B, K*2, H, W).
        seg_gt: (B, H, W) {0,1}, 1이면 물체(전경) 픽셀. 배경 픽셀은 GT 벡터가
            정의되지 않으므로(어느 키포인트로도 향할 이유가 없음) loss에서
            제외한다.
    """
    mask = seg_gt.unsqueeze(1).float()  # (B,1,H,W) - 브로드캐스트로 전체 채널에 적용
    diff = vertex_pred - vertex_gt
    loss_map = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    loss_map = loss_map * mask
    denom = mask.sum().clamp_min(1.0) * vertex_pred.shape[1]
    return loss_map.sum() / denom


def segmentation_loss(seg_logits: torch.Tensor, seg_gt: torch.Tensor) -> torch.Tensor:
    """세그멘테이션 cross-entropy loss (논문 Sec 4.1: softmax CE)."""
    return F.cross_entropy(seg_logits, seg_gt.long())