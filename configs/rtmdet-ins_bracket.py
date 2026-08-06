# =============================================================================
# RTMDet-Ins fine-tuning config for bracket detection
# =============================================================================
# Base: rtmdet-ins_tiny_8xb32-300e_coco (COCO pretrained)
# Target: 1 class (bracket), 10 images, RTX 3060
#
# 사용:
#   conda activate <본인 환경명>
#   cd <프로젝트 루트>
#   python scripts/2_Train_rtmdet_model.py --dataset <세션폴더명>
#
# 참고:
#   - MMDetection docs: https://mmdetection.readthedocs.io/en/latest/user_guides/finetune.html
#   - RTMDet paper: https://arxiv.org/abs/2212.07784
# =============================================================================

# 기본 config 상속 (모델 구조, optimizer 등)
#
# 주의: mmengine의 Config.fromfile()은 _base_ 값을 실제로 파이썬을 실행해서
# 얻는 게 아니라 AST(정적 분석)로 파일을 읽어서 상속 체인을 찾는다. 그래서
# _base_는 반드시 "리터럴 문자열"이어야 하고, os.path.join(...) 같은 계산식을
# 쓰면 mmengine이 이 config를 다른 종류(lazy_import)로 오인해서
# "ConfigParsingError: inheritance chain must match..." 오류가 난다.
#
# 그래서 여기는 (포터블하게 만들고 싶어도) 실제 환경의 mmdet 설치 경로를
# 리터럴 문자열로 그대로 적어야 한다. conda 환경 이름/사용자명이 바뀌면
# 이 줄을 직접 고쳐야 한다 (자동 탐색으로 대체할 수 없는 mmengine의 제약).
_base_ = '/home/silver/miniconda3/envs/vision_env/lib/python3.10/site-packages/mmdet/.mim/configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py'


# -----------------------------------------------------------------------------
# 1. 모델 헤드 클래스 수 변경: 80 -> 1
# -----------------------------------------------------------------------------
# RTMDet-Ins의 헤드는 클래스 개수에 의존. 1 클래스로 교체.
model = dict(
    bbox_head=dict(
        num_classes=1,  # bracket 한 개
    ),
    # 학습 시작 시 COCO pretrained 가중치 로드
    # (test_cfg는 base와 동일하게 유지)
)

# -----------------------------------------------------------------------------
# 2. 데이터셋 설정
# -----------------------------------------------------------------------------
# 학습용 데이터셋 경로
# 주의: 2_Train_rtmdet_model.py로 실행하면 --dataset 값으로 이 값이 런타임에
# override되므로, 이 config를 직접 mmdet 학습 명령으로 돌릴 때만 실제로 쓰인다.
# 프로젝트 루트 기준 상대경로 (스크립트가 os.chdir(ROOT)로 이동한 뒤 실행되므로 안전)
data_root = 'data/dataset/20260520_193909/'

# 클래스 이름과 색상 (시각화용)
metainfo = dict(
    classes=('bracket',),
    palette=[(220, 20, 60)],  # 빨강
)

# 학습 데이터셋
train_dataloader = dict(
    batch_size=2,           # RTX 3060 + 10장: 2로 시작 (메모리 안전)
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_Train.json',
        data_prefix=dict(img='intensity/'),  # PNG 위치
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

# 검증 데이터셋 (10장으론 train/val 분리 의미 없으므로 train과 동일하게)
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

# 테스트도 동일
test_dataloader = val_dataloader

# 평가 메트릭
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_Train.json',
    metric=['bbox', 'segm'],   # detection + instance segmentation 둘 다
    format_only=False,
)
test_evaluator = val_evaluator


# -----------------------------------------------------------------------------
# 3. 학습 스케줄
# -----------------------------------------------------------------------------
# 10장이라 짧게: 50 epoch
# 진짜 데이터 200~500장이면 100~300 epoch가 표준
max_epochs = 50

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10,    # 매 10 epoch마다 validation 실행
)

# 학습률 — fine-tuning이라 base보다 훨씬 낮춤
# 일반적으로 pretrained 학습률의 1/10 ~ 1/100
base_lr = 0.0001  # base config의 0.004 대비 1/4

optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
)

# Learning rate scheduler — cosine annealing
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=100,   # warmup: 처음 100 iteration
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
# 4. 출력 디렉토리 (프로젝트 루트 기준 상대경로 - os.chdir(ROOT) 이후 실행되므로 안전)
# -----------------------------------------------------------------------------
work_dir = 'work_dirs/rtmdet-ins_bracket_v1'


# -----------------------------------------------------------------------------
# 5. 체크포인트 및 로그
# -----------------------------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,           # 매 10 epoch마다 체크포인트 저장
        max_keep_ckpts=3,      # 최대 3개만 유지 (디스크 절약)
        save_best='auto',      # best 모델 자동 추적
    ),
    logger=dict(
        type='LoggerHook',
        interval=5,            # 매 5 iteration마다 로그
    ),
)


# -----------------------------------------------------------------------------
# 6. Pretrained 가중치 로드 (프로젝트 루트 기준 상대경로)
# -----------------------------------------------------------------------------
# COCO pretrained .pth 파일을 <프로젝트 루트>/models/ 에 받아두세요.
# load_from을 지정하면 mmdet이 자동으로 호환되는 레이어만 로드하고
# 클래스 수가 바뀐 head의 마지막 레이어는 새로 초기화함
load_from = 'models/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'


# -----------------------------------------------------------------------------
# 7. 자동 평균화된 학습률 조정 (auto_scale_lr)
# -----------------------------------------------------------------------------
# base config가 8 GPUs * 32 batch = 256 effective batch로 학습됨
# 우리는 1 GPU * 2 batch = 2이므로 학습률을 자동 스케일링
auto_scale_lr = dict(enable=False, base_batch_size=256)
