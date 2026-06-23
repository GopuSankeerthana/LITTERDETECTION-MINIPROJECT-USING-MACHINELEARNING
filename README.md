# Litter Detection

Detect litter in street/outdoor images using a trained Faster R-CNN model (TensorFlow).

## Quick start

```bash
# 1. One-time setup (creates venv, installs deps, downloads model ~396 MB)
bash setup.sh

# 2. Start the web app
bash run.sh
```

Open **http://127.0.0.1:7860** in your browser, upload an image, and click **Detect Litter**.

## What you get

- Green bounding boxes around detected litter
- Confidence score for each detection
- Total litter count
- Severity ranking: **Low** (0–2), **Medium** (3–5), **High** (6–8), **Very High** (9+)
- Adjustable confidence threshold slider

## Command-line test

```bash
./venv/bin/python detect.py
```

Runs on `test_images/image12.jpg` and saves output to `output/sample_result.jpg`.

## Project structure

| Path | Description |
|------|-------------|
| `app.py` | Gradio web UI for image upload |
| `detect.py` | Core inference engine |
| `litter_inference_graph/` | Trained model (download via `download_model.sh`) |
| `config/litter_detection_map.pbtxt` | Label map (single class: litter) |
| `test_images/` | Sample images for testing |
| `prototype_*.ipynb` | Original training/demo notebooks |

## Requirements

- macOS with Apple Silicon (or Python 3.11+)
- ~500 MB disk space for the model

## Troubleshooting

**Model download failed or file is 134 bytes**  
Run `bash download_model.sh` again. The real model is ~396 MB.

**TensorFlow AVX error on Intel Mac**  
Use the included `venv` with `tensorflow-macos`, or run `bash setup.sh` to recreate it.
