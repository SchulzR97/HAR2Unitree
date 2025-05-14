
# 1 Installation
> Please use Python 3.10

## 1.1 Requirements
```bash
pip install -r requirements.txt
```

## 1.2 UniTree Legged SDK
**Clone SDK**
```bash
git clone https://github.com/unitreerobotics/unitree_legged_sdk.git
cd unitree_legged_sdk
```
**Install MessagePack for C++**
```bash
sudo apt-get update
sudo apt-get install libmsgpack-dev
```
**Compile**
```bash
mkdir build && cd build
cmake -DPYTHON_BUILD=TRUE ..
make -j$(nproc)
```

**Rename File**
- Goto `lib/python/amd64`
- Copy `robot_interface.cpython-310-x86_64-linux-gnu.so` and rename it to `robot_interface.so`

**Set path in code**
- Replace the path for `ROBOT_INTERFACE_DIR` at `modules/unitree.py` with the absolute path your `lib/python/amd64` directory

## 1.3 OpenCV
```bash
$ sudo apt-get install python3-opencv
```

## 1.4 Errors

### 1.4.1 librealsense2 - Permission denied
**Error Message:**<br>
- RuntimeError: Failed to open scan_element /sys/devices/pci0000:00/0000:00:14.0/usb4/4-1/4-1:1.5/0003:8086:0B3A.0005/HID-SENSOR-200073.2.auto/iio:device1/scan_elements/in_accel_y_en Last Error: Permission denied

**How to fix this error:**<br>
- Download librealsense udev rules to allow usb device access without having sudo permissions: [https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules](https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules)
- Go to containing directory (e.g. downloads) and open a new console tab
- Copy the file to appropriate directory
    ```bash
    sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
    ```
- Reload the udev rules
    ```bash
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```

# 2 Framework
## 2.1 Human Action Recognition
## 2.2 Robot Interaction
## 2.3 PersonTracking