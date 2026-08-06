# =============================================================================
# RTMDet-Ins fine-tuning config (자동 생성 템플릿)
# =============================================================================
# scripts/make_training_config.py 로 이 템플릿을 채워서 생성한다.
# __로 감싼 토큰들이 실제 값으로 치환된다. 직접 손으로 고쳐써도 되지만,
# 이 파일 자체(_template_rtmdet.py)는 건드리지 않는 것을 권장한다
# (다른 모든 자동 생성 config가 여기서 파생되기 때문).
#
# 참고:
#   - MMDetection docs: https://mmdetection.readthedocs.io/en/latest/user_guides/finetune.html
#   - RTMDet paper: https://arxiv.org/abs/2212.07784
# =============================================================================

# mmengine의 Config.fromfile()은 _base_를 AST(정적 분석)로 읽기 때문에
# 반드시 "리터럴 문자열"이어야 한다 (계산식 불가). 환경(conda 환경 이름 등)이
# 바뀌면 scripts/make_training_config.py 실행 시 --base-config-path로 바꿔주면 된다.
_base_ = '/home/silver/miniconda3/envs/vision_env/lib/python3.10/site-packages/mmdet/.mim/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py'


# -----------------------------------------------------------------------------
# 1. 모델 헤드 클래스 수 (COCO 어노테이션의 categories 개수와 자동 일치)
# -----------------------------------------------------------------------------
model = dict(
    bbox_head=dict(
        num_classes=1,
    ),
)

# -----------------------------------------------------------------------------
# 2. 데이터셋 설정
# -----------------------------------------------------------------------------
# 주의: 2_Train_rtmdet_model.py로 실행하면 --dataset 값으로 이 값이 런타임에
# override되므로, 이 config를 직접 mmdet 학습 명령으로 돌릴 때만 실제로 쓰인다.
data_root = 'data/dataset/20260720_112039/'

# COCO 어노테이션의 categories를 id 순으로 그대로 옮긴 것.
metainfo = dict(
    classes=('bolt_m10_80',),
    palette=[(220, 20, 60)],
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_Train.json',
        data_prefix=dict(img='intensity/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

# 데이터가 적을 때는 val/test도 train과 동일하게 씀 (분리 의미 없음).
# 데이터가 충분히 쌓이면 별도 val 세션을 만들어 나눠서 지정하는 걸 권장.
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_Train.json',
        data_prefix=dict(img='intensity/'),
        test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_Train.json',
    metric=['bbox', 'segm'],
    format_only=False,
)
test_evaluator = val_evaluator


# -----------------------------------------------------------------------------
# 3. 학습 스케줄
# -----------------------------------------------------------------------------
max_epochs = 50

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,
)

base_lr = 0.0001

optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=100,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]


# -----------------------------------------------------------------------------
# 4. 출력 디렉토리
# -----------------------------------------------------------------------------
# 클래스 조합/데이터셋 이름 기반으로 자동 생성됨 - 기존 학습 결과와 절대 겹치지 않음
# (다른 객체 학습이 엉뚱하게 이전 체크포인트에서 "이어서" 시작되는 사고 방지).
work_dir = 'work_dirs/rtmdet-ins_bolt_m10_80_v1'


# -----------------------------------------------------------------------------
# 5. 체크포인트 및 로그
# -----------------------------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='auto',
    ),
    logger=dict(
        type='LoggerHook',
        interval=5,
    ),
)


# -----------------------------------------------------------------------------
# 6. Pretrained 가중치 로드
# -----------------------------------------------------------------------------
load_from = 'models/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'


# -----------------------------------------------------------------------------
# 7. 자동 학습률 스케일링
# -----------------------------------------------------------------------------
auto_scale_lr = dict(enable=False, base_batch_size=256)