# Kinecat

# Setup Guide: Kinect v2 (Xbox One) on macOS — Setup with Conda, libfreenect2, OpenCV, Ultralytics

This README sets up a **macOS** environment to stream **Kinect v2** color/depth via **libfreenect2** and use it in Python with **OpenCV** and **Ultralytics (YOLO)**.

> Microsoft’s Kinect SDK 2.0 is Windows-only; on macOS we use the open-source libfreenect2 driver + Python bindings.

---

## Hardware checklist

* Kinect for **Xbox One (v2)** sensor
* **Kinect v2 adapter** (12 V power brick + USB 3.0 breakout)
* USB-C/USB-A **USB 3.x** port (a *powered* hub is recommended if your dongle is under-powered)

---

## 0) macOS prerequisites

```bash
# Xcode command line tools (compilers, make, etc.)
xcode-select --install

# Homebrew (if you don't already have it)
# https://brew.sh has the official command; typical install:
 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew update
```

> **Apple Silicon (M-series)** Homebrew prefix is `/opt/homebrew`.
> **Intel** Macs usually use `/usr/local`.

---

## 1) Build & install libfreenect2 (the Kinect v2 driver)

```bash
# Dependencies for building libfreenect2
brew install libusb glfw cmake pkg-config jpeg-turbo

# Get the source
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2
mkdir build && cd build

# Configure (OpenGL/OpenCL optional but faster than CPU)
# NOTE: If you hit a "CMake < 3.5 policy" error, see the comment below.
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENGL=ON \
  -DENABLE_OPENCL=ON

# If you get a CMake policy error:
# cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENGL=ON -DENABLE_OPENCL=ON

make -j
sudo make install
```

**Test the driver:**

```bash
# From libfreenect2/build (or wherever 'Protonect' was installed)
./bin/Protonect
# You should see live Color/IR/Depth windows. CTRL+C to quit.
```

---

## 2) Create the Conda environment

```bash
# Create and activate a clean environment
conda create -n kinect python=3.10 -y
conda activate kinect

# Use conda-forge with strict priority to avoid mixed binaries
conda config --add channels conda-forge
conda config --set channel_priority strict
```

**Install OpenCV + BLAS/LAPACK via conda (binary-compatible):**

```bash
# Avoid pip OpenCV wheels inside conda; use conda-forge builds
conda install -y opencv numpy libopenblas liblapack
```

**Install pip-only packages:**

```bash
# YOLO (Ultralytics) and the Python bindings for libfreenect2
pip install ultralytics pylibfreenect2
```

> If you previously installed any `opencv-python*` wheels via pip, remove them to avoid `dlopen` conflicts:
>
> ```bash
> pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
> ```

---

## 3) Point Python to the libfreenect2 native libraries

Set these **each time** before running Python (you can add them to your shell profile):

```bash
# Apple Silicon (Homebrew default prefix)
export LIBFREENECT2_INSTALL_PREFIX=/opt/homebrew
# Intel Macs:
# export LIBFREENECT2_INSTALL_PREFIX=/usr/local

# Make sure the dynamic linker can find the libfreenect2 dylibs
export DYLD_LIBRARY_PATH="$LIBFREENECT2_INSTALL_PREFIX/lib:$DYLD_LIBRARY_PATH"
```

**(Optional) Detect your arch & set automatically**

```bash
if [ "$(uname -m)" = "arm64" ]; then
  export LIBFREENECT2_INSTALL_PREFIX=/opt/homebrew
else
  export LIBFREENECT2_INSTALL_PREFIX=/usr/local
fi
export DYLD_LIBRARY_PATH="$LIBFREENECT2_INSTALL_PREFIX/lib:$DYLD_LIBRARY_PATH"
```

---

## 4) Verify the Python stack

```bash
# In the conda 'kinect' env, with the DYLD vars set:
python - <<'PY'
import platform, cv2
print("Arch:", platform.machine())
print("OpenCV:", cv2.__version__)
try:
    import pylibfreenect2
    print("pylibfreenect2: OK")
except Exception as e:
    print("pylibfreenect2 import error:", e)
try:
    import ultralytics
    print("ultralytics: OK")
except Exception as e:
    print("ultralytics import error:", e)
PY
```

If imports succeed, you’re ready to run your Kinect+YOLO application.

---

## Troubleshooting

* **CMake policy error during build**
  Use `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` in your `cmake ..` command, or edit the project’s `CMakeLists.txt` to `cmake_minimum_required(VERSION 3.5)` and reconfigure from a clean `build/`.

* **`Protonect` can’t see the sensor**
  Ensure the Kinect v2 adapter is powered (12 V) and you’re on a **USB 3.x** port. Prefer a powered USB-C hub. Try a different cable/port.

* **OpenCV import error like `liblapack.3.dylib not found`**
  You’re mixing pip wheels with conda libraries. In your conda env:

  ```bash
  pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
  conda install -c conda-forge opencv libopenblas liblapack
  ```

* **`pylibfreenect2` import error / library not loaded**
  Confirm:

  ```bash
  echo $LIBFREENECT2_INSTALL_PREFIX
  ls "$LIBFREENECT2_INSTALL_PREFIX/lib" | grep freenect2
  echo $DYLD_LIBRARY_PATH
  ```

  The `libfreenect2*.dylib` files must be in a directory listed in `DYLD_LIBRARY_PATH`.

* **Apple Silicon arch mismatch**
  Keep everything **arm64** (Homebrew under `/opt/homebrew`, Python `arm64`). Mixing x86_64 Python with arm64 libs (or vice versa) will fail.

* **OpenGL/OpenCL pipeline issues**
  Rebuild libfreenect2 with `-DENABLE_OPENGL=OFF -DENABLE_OPENCL=OFF` to fall back to CPU (slower but reliable).

* **Depth alignment later**
  libfreenect2 provides registration utilities to map depth↔color. Add `FrameType.Depth` in your listener when you need it.

---

## Optional: reproducible `environment.yml`

```yaml
name: kinect
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - opencv
  - libopenblas
  - liblapack
  - pip
  - pip:
      - ultralytics
      - pylibfreenect2
```

Create it with:

```bash
conda env create -f environment.yml
conda activate kinect
```

> Remember to export the **`LIBFREENECT2_INSTALL_PREFIX`** and **`DYLD_LIBRARY_PATH`** before running Python.






---


Here’s a **copy-pasteable README section** for a clean macOS setup that gets **Ultralytics YOLO + OpenCV + Kinect v2 (libfreenect2/pylibfreenect2)** working in a Conda env, with a webcam fallback.

````markdown
# Environment Setup (macOS, Conda) — YOLO + Kinect v2 + OpenCV

These steps set up a Python environment that runs Ultralytics YOLO with frames from **Kinect v2** (via `libfreenect2`/`pylibfreenect2`) and falls back to a **webcam** if Kinect isn’t present.

> Tested on macOS with Conda (Python 3.11).  
> If you only want YOLO + webcam, stop after Step 3.

---

## 0) Prerequisites

Install Xcode command line tools and Homebrew:
```bash
xcode-select --install || true
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
````

Install build/runtime dependencies:

```bash
brew update
brew install cmake libusb glfw pkg-config ffmpeg git
```

---

## 1) Create a clean Conda environment

```bash
# Keep conda-forge consistent and fast
conda config --set channel_priority strict
conda config --add channels conda-forge

# New environment (Python 3.11 recommended)
conda create -n vision python=3.11 -y
conda activate vision

# Optional (faster solver)
conda install -n base -c conda-forge mamba -y
```

---

## 2) Install core Python packages

```bash
# OpenCV (from conda-forge for correct native deps)
conda install -y opencv ffmpeg

# Ultralytics from pip (their conda build lags)
pip install --upgrade pip setuptools wheel
pip install ultralytics
```

> **PyTorch note:** Ultralytics installs a suitable PyTorch automatically. On macOS there’s no CUDA; CPU-only is fine by default. If you need to pin PyTorch versions, do it before installing `ultralytics`.

Quick sanity check:

```bash
python - << 'PY'
import cv2
print("OpenCV:", cv2.__version__)
from ultralytics import YOLO
print("Ultralytics import OK")
PY
```

---

## 3) (Optional) YOLO model file

Ultralytics will auto-download `yolo11n.pt`. If you prefer manual:

```bash
# Place the model file next to your script
# (Or just let ultralytics download on first run)
```

---

## 4) Install libfreenect2 (Kinect v2 driver)

> **Only needed if you will use Kinect v2.**

Build and install to a user prefix (easier to manage):

```bash
git clone https://github.com/OpenKinect/libfreenect2.git
cd libfreenect2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$HOME/freenect2"
make -j"$(sysctl -n hw.logicalcpu)"
make install
```

Confirm dylibs exist:

```bash
ls -l "$HOME/freenect2/lib"/libfreenect2*.dylib
```

---

## 5) Install pylibfreenect2 (Python bindings)

```bash
cd ~
pip uninstall -y pylibfreenect2 || true
pip install git+https://github.com/r9y9/pylibfreenect2.git
```

---

## 6) Make sure macOS can find the libfreenect2 dylib

Choose **one** of the options below.

### Option A — Add a Conda activation hook (clean)

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/freenect2.sh" <<'EOF'
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/freenect2/lib:${DYLD_FALLBACK_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$HOME/freenect2/lib/pkgconfig:${PKG_CONFIG_PATH}"
EOF
# Re-activate to apply
conda deactivate && conda activate vision
```

### Option B — Symlink the dylib into the env (quick)

```bash
ln -sf "$HOME/freenect2/lib/libfreenect2.0.2.dylib" "$CONDA_PREFIX/lib/libfreenect2.0.2.dylib"
ln -sf "$HOME/freenect2/lib/libfreenect2.dylib"      "$CONDA_PREFIX/lib/libfreenect2.dylib"
```

---

## 7) Verify Kinect detection (optional)

Plug in Kinect v2 and run:

```bash
python - << 'PY'
from pylibfreenect2 import Freenect2
fn = Freenect2()
print("Devices found:", fn.enumerateDevices())
if fn.enumerateDevices() > 0:
    print("Serial 0:", fn.getDeviceSerialNumber(0))
PY
```

If you see a nonzero device count and a serial number, libfreenect2 is good.
