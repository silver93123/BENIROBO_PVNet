"""Femto Bolt가 실제로 지원하는 스트림 프로파일(해상도/포맷/fps) 덤프.

configs/camera_config_femto.yaml의 depth_width/depth_height/fps를 바꾸기 전에
이 스크립트로 먼저 이 기기가 실제로 지원하는 조합을 확인하세요. 임의의 조합을
넣으면 촬영 시 "Invalid input, No matched video stream profile found!" 오류가
납니다 (예: fps=8은 대부분의 조합에서 미지원 - Femto Bolt는 보통 5/15/30만
지원합니다).

참고: src/camera/femto_bolt.py의 open()도 이제 프로파일을 못 찾으면 이 스크립트와
같은 방식으로 지원 목록을 에러 메시지에 함께 출력하므로, 급하면 그냥 한 번
촬영을 시도해서 에러 메시지만 봐도 됩니다. 이 스크립트는 촬영 시도 없이 미리
전체 목록을 훑어보고 싶을 때 씁니다.

사용법:
    python scripts/list_femto_profiles.py
"""
from __future__ import annotations

try:
    from pyorbbecsdk import OBSensorType, Pipeline
except ImportError as e:
    raise SystemExit(
        "pyorbbecsdk를 import할 수 없습니다. 'pip install pyorbbecsdk2' 실행 후 다시 시도하세요. "
        f"원본 에러: {e}"
    )

SENSORS = {
    "DEPTH": OBSensorType.DEPTH_SENSOR,
    "IR": OBSensorType.IR_SENSOR,
    "COLOR": OBSensorType.COLOR_SENSOR,
}


def main() -> None:
    pipeline = Pipeline()
    try:
        device_info = pipeline.get_device().get_device_info()
        print(f"연결된 기기: {device_info.get_name()} (S/N={device_info.get_serial_number()}, "
              f"FW={device_info.get_firmware_version()})\n")
    except Exception as e:
        print(f"[경고] 기기 정보를 가져오지 못했습니다: {e}\n")

    for label, sensor_type in SENSORS.items():
        print(f"=== {label} 지원 프로파일 ===")
        try:
            profile_list = pipeline.get_stream_profile_list(sensor_type)
        except Exception as e:
            print(f"  (조회 실패: {e})\n")
            continue

        if profile_list is None or profile_list.get_count() == 0:
            print("  (지원하는 프로파일이 없습니다 - 이 기기에 해당 센서가 없을 수 있습니다)\n")
            continue

        seen = set()
        for i in range(profile_list.get_count()):
            p = profile_list.get_stream_profile_by_index(i)
            line = f"{p.get_width()}x{p.get_height()} @ {p.get_fps()}fps  format={p.get_format()}"
            if line not in seen:
                seen.add(line)
                print(f"  {line}")
        print()

    pipeline.stop()


if __name__ == "__main__":
    main()