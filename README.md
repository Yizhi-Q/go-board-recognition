# Go Board Recognition

English | [简体中文](README.zh-CN.md)

An end-to-end computer vision application for reconstructing Go games from video. The project detects stones on a 19×19 board, tracks board states over time, and exports the recognised game to Smart Game Format (SGF).

Developed as an undergraduate Computer Science capstone project.

## Features

- Desktop user interface built with Tkinter
- Video-frame extraction and stability filtering
- Perspective correction for board images
- YOLO-based black/white stone detection
- Mapping detections to a 19×19 grid and tracking board states
- Optional hand-occlusion filtering
- SGF export and reconstructed-board image generation

## Project workflow

```text
Input video → stable-frame selection → perspective correction → stone detection
→ grid mapping → board-state tracking → SGF / board-image export
```

## Demo

### Desktop application workflow

The interface displays the reconstructed board, generated SGF content, move history, and processing controls in one workspace.

![Desktop application workflow](docs/assets/application-workflow.png)

### Board reconstruction

The application maps detected stones from a physical Go board image to positions on a 19×19 digital board.

![Input board and reconstructed board](docs/assets/board-reconstruction.png)

## Repository structure

```text
.
├── main.py                    # Application entry point
├── GUI.py                     # Tkinter interface
├── GBR.py                     # Video-processing workflow
├── Chess_Recognition/         # Detection, grid mapping, SGF and game-state logic
├── Preprocess/                # Frame extraction and perspective correction
├── Test_Image/                # Sample images
├── docs/assets/                # Demo screenshots
├── yaml/                      # Experimented YOLO model configurations
└── models/                    # Local model checkpoints (not committed)
```

## Setup

1. Create and activate a Python virtual environment (Python 3.10+ recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the stone-detection model at `models/best_300epoch_onlybw.pt`.
4. Optionally place the hand-occlusion model at `models/hand_s.pt`.
5. Start the desktop application:

   ```bash
   python main.py
   ```

The application uses CUDA when available and otherwise falls back to CPU.

## Notes

- Source-code comments and user-interface text are primarily in Chinese; documentation is available in English and Simplified Chinese.
- Model weights and generated outputs are intentionally excluded from version control.
- The YAML files record model-architecture experiments. Check the applicable third-party licences before reusing or distributing derived configurations or weights.

## Future improvements

- Add a small labelled evaluation set and report detection / board-state accuracy.
- Add automated tests for grid mapping and SGF generation.
- Provide a short demonstration video covering video input through SGF export.
