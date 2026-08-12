"""
Orbbec Femto Bolt: 디바이스 프로퍼티(OBPropertyID) 조회/토글 유틸리티

핵심 아이디어:
  pyorbbecsdk는 기기가 지원하지 않는 프로퍼티를 get/set 하면 OBError
  (내부적으로 OB_EXCEPTION_TYPE_UNSUPPORTED_OPERATION 등)를 던집니다.
  이 스크립트는 "지원 여부를 미리 알려주는 전용 API"에 의존하는 대신,
  실제로 get_xxx_property()를 호출해보고 성공하면 지원, 예외가 나면
  미지원으로 판단하는 방식으로 지원 여부를 자동 체크합니다.
  (SDK/펌웨어 버전에 따라 지원 프로퍼티 목록이 달라지므로, 이 방식이
  가장 안전하고 버전에 덜 민감합니다.)

참고 문서/소스 (공식, 확인됨):
- https://github.com/orbbec/pyorbbecsdk
  (docs/README_CN.md, README_EN.md 의 device.set_bool_property /
   get_int_property / set_int_property 사용 패턴)
- OBPropertyID 목록: OrbbecSDK v2 헤더 기반 공개 문서
  (https://docs.rs/orbbec-sdk-sys — OrbbecSDK_v2 C 헤더의 Rust 바인딩 문서,
   프로퍼티 이름과 설명이 원본 SDK 주석 그대로 노출되어 있어 이름 확인용으로 사용)
- OBExceptionType(OB_EXCEPTION_TYPE_UNSUPPORTED_OPERATION 등)도 위 문서에서 확인

주의:
- 아래 PROPERTIES 목록은 OrbbecSDK 전체(구조광/스테레오/ToF 기종 포함) 기준입니다.
  레이저 관련 프로퍼티 상당수는 구조광 카메라용이라 ToF 방식인 Femto Bolt에서는
  미지원(NOT SUPPORTED)으로 나올 수 있습니다 — 이는 정상입니다.
- 실제 지원 여부/현재값/설정 가능 여부는 반드시 이 스크립트로 실기에서
  확인하시길 권장합니다.

필요 패키지:
    pip install pyorbbecsdk

사용법:
    python femto_bolt_property_util.py --list                       # 지원 프로퍼티 전체 조회
    python femto_bolt_property_util.py --list --category color      # 카테고리별 조회
    python femto_bolt_property_util.py --get OB_PROP_COLOR_EXPOSURE_INT
    python femto_bolt_property_util.py --set OB_PROP_COLOR_EXPOSURE_INT 200
    python femto_bolt_property_util.py --toggle OB_PROP_LASER_BOOL    # bool 프로퍼티 on/off 반전
"""

import argparse
from pyorbbecsdk import *


# ============================================================
# 프로퍼티 레지스트리
# 이름은 OBPropertyID enum 멤버명과 동일하게 맞춰 getattr(OBPropertyID, name)로 사용합니다.
# type: "bool" | "int" | "float"
# ============================================================
PROPERTIES = {
    "color": {
        "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL": "bool",
        "OB_PROP_COLOR_EXPOSURE_INT": "int",
        "OB_PROP_COLOR_GAIN_INT": "int",
        "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL": "bool",
        "OB_PROP_COLOR_WHITE_BALANCE_INT": "int",
        "OB_PROP_COLOR_BRIGHTNESS_INT": "int",
        "OB_PROP_COLOR_CONTRAST_INT": "int",
        "OB_PROP_COLOR_SATURATION_INT": "int",
        "OB_PROP_COLOR_SHARPNESS_INT": "int",
        "OB_PROP_COLOR_GAMMA_INT": "int",
        "OB_PROP_COLOR_HUE_INT": "int",
        "OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT": "int",
        "OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT": "int",
        "OB_PROP_COLOR_MIRROR_BOOL": "bool",
        "OB_PROP_COLOR_FLIP_BOOL": "bool",
        "OB_PROP_COLOR_ROTATE_INT": "int",
        "OB_PROP_COLOR_HDR_BOOL": "bool",
    },
    "depth": {
        "OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL": "bool",
        "OB_PROP_DEPTH_EXPOSURE_INT": "int",
        "OB_PROP_DEPTH_GAIN_INT": "int",
        "OB_PROP_MIN_DEPTH_INT": "int",
        "OB_PROP_MAX_DEPTH_INT": "int",
        "OB_PROP_DEPTH_MIRROR_BOOL": "bool",
        "OB_PROP_DEPTH_FLIP_BOOL": "bool",
        "OB_PROP_DEPTH_ROTATE_INT": "int",
        "OB_PROP_DEPTH_PRECISION_LEVEL_INT": "int",
        "OB_PROP_DEPTH_UNIT_FLEXIBLE_ADJUSTMENT_FLOAT": "float",
        "OB_PROP_DEPTH_HOLEFILTER_BOOL": "bool",
        "OB_PROP_DEPTH_SOFT_FILTER_BOOL": "bool",
        "OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL": "bool",
        "OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_DIFF_INT": "int",
        "OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_SPECKLE_SIZE_INT": "int",
        "OB_PROP_HDR_MERGE_BOOL": "bool",
    },
    "ir": {
        "OB_PROP_IR_AUTO_EXPOSURE_BOOL": "bool",
        "OB_PROP_IR_EXPOSURE_INT": "int",
        "OB_PROP_IR_GAIN_INT": "int",
        "OB_PROP_IR_BRIGHTNESS_INT": "int",
        "OB_PROP_IR_MIRROR_BOOL": "bool",
        "OB_PROP_IR_FLIP_BOOL": "bool",
        "OB_PROP_IR_ROTATE_INT": "int",
    },
    "align": {
        "OB_PROP_DEPTH_ALIGN_HARDWARE_BOOL": "bool",
        "OB_PROP_D2C_PREPROCESS_BOOL": "bool",
        "OB_PROP_RGB_CUSTOM_CROP_BOOL": "bool",
    },
    "device": {
        "OB_PROP_LASER_BOOL": "bool",
        "OB_PROP_LDP_BOOL": "bool",
        "OB_PROP_INDICATOR_LIGHT_BOOL": "bool",
        "OB_PROP_FAN_WORK_MODE_INT": "int",
        "OB_PROP_DEVICE_REBOOT_DELAY_INT": "int",
    },
}


def get_device():
    """Pipeline을 통해 첫 번째 연결된 디바이스 핸들을 얻는다."""
    pipeline = Pipeline()
    device = pipeline.get_device()
    return pipeline, device


def read_property(device, name: str, ptype: str):
    """프로퍼티를 읽어본다. 성공하면 (True, 값), 실패(미지원 등)하면 (False, None)."""
    prop_id = getattr(OBPropertyID, name, None)
    if prop_id is None:
        return False, None  # 이 SDK 버전에 해당 enum 멤버 자체가 없음
    try:
        if ptype == "bool":
            return True, device.get_bool_property(prop_id)
        elif ptype == "int":
            return True, device.get_int_property(prop_id)
        elif ptype == "float":
            return True, device.get_float_property(prop_id)
    except OBError:
        return False, None
    return False, None


def write_property(device, name: str, ptype: str, value):
    prop_id = getattr(OBPropertyID, name, None)
    if prop_id is None:
        print(f"[실패] {name}: 이 SDK 버전에 존재하지 않는 프로퍼티입니다.")
        return False
    try:
        if ptype == "bool":
            device.set_bool_property(prop_id, bool(value))
        elif ptype == "int":
            device.set_int_property(prop_id, int(value))
        elif ptype == "float":
            device.set_float_property(prop_id, float(value))
        print(f"[성공] {name} = {value} 로 설정했습니다.")
        return True
    except OBError as e:
        print(f"[실패] {name} 설정 불가 (미지원이거나 범위를 벗어남): {e}")
        return False


def list_properties(device, category: str = None):
    categories = {category: PROPERTIES[category]} if category else PROPERTIES
    for cat_name, props in categories.items():
        print(f"\n=== {cat_name.upper()} ===")
        for name, ptype in props.items():
            ok, value = read_property(device, name, ptype)
            if ok:
                print(f"  [지원 O] {name:<55} ({ptype:5s}) 현재값 = {value}")
            else:
                print(f"  [지원 X] {name:<55} ({ptype:5s}) -- 이 기기/SDK에서 미지원")


def find_property_type(name: str):
    for props in PROPERTIES.values():
        if name in props:
            return props[name]
    return None


def main():
    parser = argparse.ArgumentParser(description="Femto Bolt 디바이스 프로퍼티 조회/토글 유틸리티")
    parser.add_argument("--list", action="store_true", help="프로퍼티 지원 여부 + 현재값 전체 조회")
    parser.add_argument("--category", choices=list(PROPERTIES.keys()), help="--list 결과를 특정 카테고리로 제한")
    parser.add_argument("--get", metavar="PROPERTY_NAME", help="특정 프로퍼티 값 조회")
    parser.add_argument("--set", nargs=2, metavar=("PROPERTY_NAME", "VALUE"), help="특정 프로퍼티 값 설정")
    parser.add_argument("--toggle", metavar="PROPERTY_NAME", help="bool 프로퍼티 현재값을 반전(on<->off)")
    args = parser.parse_args()

    if not any([args.list, args.get, args.set, args.toggle]):
        parser.print_help()
        return

    pipeline, device = get_device()

    try:
        if args.list:
            list_properties(device, args.category)

        if args.get:
            ptype = find_property_type(args.get)
            if ptype is None:
                print(f"'{args.get}'는 레지스트리에 없는 이름입니다. --list로 사용 가능한 이름을 확인하세요.")
            else:
                ok, value = read_property(device, args.get, ptype)
                if ok:
                    print(f"{args.get} = {value}")
                else:
                    print(f"{args.get}: 이 기기/SDK에서 미지원이거나 조회 실패")

        if args.set:
            name, raw_value = args.set
            ptype = find_property_type(name)
            if ptype is None:
                print(f"'{name}'는 레지스트리에 없는 이름입니다. --list로 사용 가능한 이름을 확인하세요.")
            else:
                if ptype == "bool":
                    value = raw_value.lower() in ("1", "true", "on", "yes")
                elif ptype == "int":
                    value = int(raw_value)
                else:
                    value = float(raw_value)
                write_property(device, name, ptype, value)

        if args.toggle:
            name = args.toggle
            ptype = find_property_type(name)
            if ptype != "bool":
                print(f"--toggle은 bool 프로퍼티에만 사용할 수 있습니다. ({name}은 {ptype})")
            else:
                ok, current = read_property(device, name, "bool")
                if not ok:
                    print(f"{name}: 이 기기/SDK에서 미지원이라 토글할 수 없습니다.")
                else:
                    write_property(device, name, "bool", not current)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()