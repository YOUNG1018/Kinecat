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

