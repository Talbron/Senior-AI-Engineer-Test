# Senior AI Engineer Test

![CI](https://github.com/Talbron/Senior-AI-Engineer-Test/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/Talbron/Senior-AI-Engineer-Test/branch/main/graph/badge.svg)](https://codecov.io/gh/Talbron/Senior-AI-Engineer-Test)
![Python](https://img.shields.io/badge/python-3.13-blue)


## Contents
- [Introduction](#introduction)
- [Structure](#structure)
- [Installation](#installation)
- [Testing](#testing)
- [Running-the-code](#running-the-code)
- [Known-Issues](#known-issues)

---

## Lab Monitor

### Introduction
This repository contains code for submission for the Senior AI Engineer test.  
A processing pipeline was developed that performs:
- Lens correction  
- Object detection of lab equipment  
- Rudimentary interaction logging  

Key technologies:
- [**OpenCV**](https://opencv.org) for video/image manipulation  
- [**Grounding DINO**](https://github.com/IDEA-Research/GroundingDINO) for zero-shot object detection  

Supporting files:
- [`PLAN.md`](https://github.com/Talbron/Senior-AI-Engineer-Test/blob/main/PLAN.md): Original notes taken when first watching the video  
- [`REPORT.md`](https://github.com/Talbron/Senior-AI-Engineer-Test/blob/main/REPORT.md): Final report and further development plans  

---

### Structure
- `legacy/`: Original forked repository  
- `src/`: Library of developed functions  
- `tests/`: Unit tests for the library functions  
- `experiments/`: Development experiments (preserved for reference)  
- `samples/`: Screenshots taken from the test video for rapid development  

---

### Installation

- A `Dockerfile` is provided and **must be used** to run DINO object detection.  
  - Recommended to use with **NVIDIA Docker** for GPU acceleration.
- `devcontainer.json` is included for integration with VSCode Devcontainers.
- Dependency management is handled via **Poetry**:
  
  ```bash
  poetry install
  ```

- All commands should be prefixed with `poetry run`, for example:

  ```bash
  poetry run python main.py
  poetry run pytest
  ```

#### 🔧 Known Setup Bug (Grounding DINO)
- **Issue**: [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) is broken on initial setup.
- A fix script is included: `dino_fix.sh`
  - It *should* be applied during the Docker build, but **currently is not**.
  - To apply the fix manually:

    ```bash
    chmod +x dino_fix.sh
    ./dino_fix.sh
    ```

  - Only needs to be run once. It will take some time to re-install DINO.

#### ❗ Special Note on GUI Scripts
Two scripts **will not run inside Docker** due to GUI limitations:
- `experiments/manual_calibration.py`
- `experiments/manual_hsv_tuner.py`

These use OpenCV GUIs for visual parameter tuning. They:
- Do **not** require DINO  
- Can be safely run outside Docker using:

  ```bash
  poetry run python <script_name>
  ```

---

### Testing

Run all tests and code quality checks using Poetry:

```bash
poetry run pytest
poetry run flake8
poetry run pylint src/lab_monitor
```

- CI pipelines are configured to run tests on the `main` branch.

---

### Running the Code

The video processing pipeline runs as an **API**:

```bash
poetry run python main.py
```

- Default port: **8001**
- To change the port, edit the bottom line of `main.py`
- Access the API documentation at: [http://127.0.0.1:8001/docs#/](http://127.0.0.1:8001/docs#/)

#### Optional Video Processing
You may also run:

```bash
poetry run python experiments/process_video.py
```

**Requirements:**
- A folder named `data/` in the root directory
- The video file `AICandidateTest-FINAL.mp4` inside `data/`

---

### Endpoints

- **Upload Video**: Upload an `.mp4`, returns a `job_id`
- **Status**: Use the `job_id` to check % complete
- **Download Video**: Retrieve the processed video
- **Download Log**: Retrieve the generated action log

---

## Known Issues

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) is broken on setup — must run `dino_fix.sh` manually.
- Action logging **should** detect:
  - Touch/release bottle  
  - Touch/release petri dish  
  - Open/close bottle  
  - Pour bottle  
  However, **only bottle/hand interactions are currently being logged**.
