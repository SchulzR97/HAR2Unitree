
# 1 Installation
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

## 1.3 OpenCV
```bash
$ sudo apt-get install python3-opencv
```

# 2 Framework
## 2.1 Human Action Recognition
## 2.2 Robot Interaction
## 2.3 PersonTracking