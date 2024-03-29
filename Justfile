export HSA_OVERRIDE_GFX_VERSION := "11.0.0"
export HIP_VISIBLE_DEVICES := "0"

export CXX := "/opt/rocm/bin/hipcc"
export CC := "/opt/rocm/bin/hipcc"
export CMAKE_ARGS := "-DLLAMA_HIPBLAS=on"
export GPU_MAX_HW_QUEUES := "1"

fetch:
	# Remove torch dependencies manually
	git clone git@github.com:salesforce/LAVIS.git text-generation/LAVIS

setup-text-generation:
	#!/usr/bin/env bash
	cd text-generation
	# Create a virtual environment to isolate dependencies
	python -m venv venv
	# Install pytorch toolchain with ROCm enabled
	venv/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7/
	# Build LAVIS from source
	venv/bin/pip install -e LAVIS
	# Install llama.cpp python bindings
	venv/bin/pip install llama-cpp-python
	# Install the FastAPI web framework for serving a single endpoint
	venv/bin/pip install fastapi "uvicorn[standard]" python-multipart

setup-speech-generation:
	#!/usr/bin/env bash
	cd speech-generation
	# Create a virtual environment to isolate dependencies
	python -m venv venv
	# Install pytorch toolchain with ROCm enabled
	venv/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7/
	venv/bin/pip install TTS
	# Install the FastAPI web framework for serving a single endpoint
	venv/bin/pip install fastapi "uvicorn[standard]" python-multipart

setup: setup-text-generation setup-speech-generation

text-generation:
	#!/usr/bin/env bash
	cd text-generation
	venv/bin/uvicorn main:app --port 8001

speech-generation $GPU_MAX_HW_QUEUES="1":
	#!/usr/bin/env bash
	cd speech-generation
	venv/bin/uvicorn main:app --port 8002

