#!/usr/bin/env bash
# Step 2b: lock white balance, exposure, gain and focus on a UVC webcam, then dump the result.
# LeRobot's OpenCV camera config has no exposure/white-balance fields, so this is done out of band.
#
#   sudo apt install v4l-utils
#   steering/lock_camera.sh /dev/v4l/by-id/usb-<cam0>-video-index0 [exposure] [wb_kelvin] [gain] [focus]
#
# Control names differ between cameras: run `v4l2-ctl -d DEV --list-ctrls` first and adjust.
# Many UVC cameras forget these on replug or reboot, and some reset them when a program opens the
# device, so re-run this after connecting, and re-check with --list-ctrls after the first lerobot
# command has opened the camera. Record the printed values in the run log.
set -euo pipefail
dev=${1:?usage: lock_camera.sh DEVICE [exposure] [wb_kelvin] [gain] [focus]}
exposure=${2:-156}
wb=${3:-4600}
gain=${4:-0}
focus=${5:-0}

try() { v4l2-ctl -d "$dev" --set-ctrl="$1" 2>/dev/null && echo "set $1" || echo "skip $1 (not supported)"; }

# Auto modes off first, otherwise the manual values are ignored.
try auto_exposure=1                 # 1 = manual on most UVC cameras (older drivers: exposure_auto=1)
try exposure_auto=1
try exposure_dynamic_framerate=0
try white_balance_automatic=0       # older drivers: white_balance_temperature_auto=0
try white_balance_temperature_auto=0
try focus_automatic_continuous=0    # older drivers: focus_auto=0
try focus_auto=0
try backlight_compensation=0
try power_line_frequency=2          # 50 Hz mains (Singapore); stops fluorescent banding

try exposure_time_absolute="$exposure"
try exposure_absolute="$exposure"
try white_balance_temperature="$wb"
try gain="$gain"
try focus_absolute="$focus"

echo "--- $dev"
v4l2-ctl -d "$dev" --list-ctrls
