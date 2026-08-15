import os
import time
from subprocess import Popen, PIPE, DEVNULL
import fcntl
import pathlib

_CAPTURE_CARD_MARKERS = ("Elgato", "Cam_Link", "HD60", "Game_Capture")


def is_capture_card(path: str) -> bool:
    return any(marker in path for marker in _CAPTURE_CARD_MARKERS)


def create_usb_list():
    device_list = list()
    lsusb_out = Popen('lsusb -v', shell=True, bufsize=64,
                      stdin=PIPE, stdout=PIPE, stderr=DEVNULL,
                      close_fds=True).stdout.read().strip().decode('utf-8')
    usb_devices = lsusb_out.split('%s%s' % (os.linesep, os.linesep))
    for device_categories in usb_devices:
        if not device_categories:
            continue
        categories = device_categories.split(os.linesep)
        device_stuff = categories[0].strip().split()
        bus = device_stuff[1]
        device = device_stuff[3][:-1]
        device_dict = {'bus': bus, 'device': device}
        device_info = ' '.join(device_stuff[6:])
        device_dict['description'] = device_info
        for category in categories:
            if not category:
                continue
            categoryinfo = category.strip().split()
            if categoryinfo[0] == 'iManufacturer':
                manufacturer_info = ' '.join(categoryinfo[2:])
                device_dict['manufacturer'] = manufacturer_info
            if categoryinfo[0] == 'iProduct':
                device_info = ' '.join(categoryinfo[2:])
                device_dict['device'] = device_info
        path = '/dev/bus/usb/%s/%s' % (bus, device)
        device_dict['path'] = path

        device_list.append(device_dict)
    return device_list


def reset_usb_device(dev_path, *, optional: bool = False) -> bool:
    USBDEVFS_RESET = 21780
    try:
        f = open(dev_path, 'w', os.O_WRONLY)
        try:
            fcntl.ioctl(f, USBDEVFS_RESET, 0)
        finally:
            f.close()
        print(f"Successfully reset {dev_path}")
        return True
    except PermissionError:
        msg = f"USB reset permission denied for {dev_path}; try: chmod 777 {dev_path}"
        if optional:
            print(f"[WARN] {msg}")
            return False
        raise PermissionError(msg)
    except OSError as exc:
        # Common in Docker: USBDEVFS_RESET is not supported (errno 25).
        msg = f"USB reset skipped for {dev_path}: {exc}"
        if optional:
            print(f"[WARN] {msg}")
            return False
        raise


def reset_all_elgato_devices():
    """
    Find and reset all Elgato capture cards.
    Required to workaround a firmware bug on bare metal; optional in Docker.
    """
    device_list = create_usb_list()
    reset_any = False
    for dev in device_list:
        if 'Elgato' in dev.get('description', ''):
            if reset_usb_device(dev['path'], optional=True):
                reset_any = True
    if not reset_any:
        print("[WARN] Elgato USB reset not performed (Docker or permissions); continuing.")


def _probe_capture(path: str) -> bool:
    try:
        import cv2

        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        deadline = time.time() + (5.0 if is_capture_card(path) else 1.0)
        while time.time() < deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                if (len(frame.shape) == 3) and (frame.shape[2] == 3):
                    cap.release()
                    return True
            time.sleep(0.1)
        cap.release()
        return False
    except Exception:
        return False


def _path_is_available(path: str) -> bool:
    """True if the v4l path exists and its device node is present (symlink target ok)."""
    p = pathlib.Path(path)
    if not p.exists():
        return False
    try:
        return p.resolve().exists()
    except OSError:
        return False


def get_sorted_v4l_paths(by_id=True):
    """
    If by_id, sort devices by device name + serial number (preserves device order)
    else, sort devices by usb bus id (preserves usb port order)
    """

    def _collect_from_v4l_dir(dirname: str):
        v4l_dir = pathlib.Path('/dev/v4l').joinpath(dirname)
        paths = list()
        for dev_path in sorted(v4l_dir.glob("*video*")):
            name = dev_path.name
            index_str = name.split('-')[-1]
            if index_str.startswith('index'):
                try:
                    index = int(index_str[5:])
                except ValueError:
                    continue
                if index == 0:
                    if _path_is_available(str(dev_path)):
                        paths.append(dev_path)
                    elif is_capture_card(str(dev_path)):
                        print(
                            f"[WARN] Capture card path exists but device node missing: {dev_path}. "
                            "In Docker run: bash docker/container.sh attach-video"
                        )
        return paths

    primary_dir = 'by-id' if by_id else 'by-path'
    valid_paths = _collect_from_v4l_dir(primary_dir)

    if len(valid_paths) == 0:
        other_dir = 'by-path' if by_id else 'by-id'
        valid_paths = _collect_from_v4l_dir(other_dir)

    if len(valid_paths) == 0:
        valid_paths = [
            p for p in sorted(pathlib.Path('/dev').glob('video*'))
            if _path_is_available(str(p))
        ]

    result = [str(x.absolute()) for x in valid_paths]
    result.sort(key=lambda p: (0 if is_capture_card(p) else 1, p))

    probed = {p for p in result if _probe_capture(p)}
    capture_cards = [p for p in result if is_capture_card(p)]

    # Always list HDMI capture cards first (even if probe fails: no GoPro HDMI yet).
    out: list[str] = []
    for p in capture_cards:
        if p not in out:
            out.append(p)
    for p in result:
        if p not in out and p in probed:
            out.append(p)

    return out if out else result
