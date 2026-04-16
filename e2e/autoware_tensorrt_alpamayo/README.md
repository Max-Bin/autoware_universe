# Autoware TensorRT Alpamayo

## Overview

**Autoware TensorRT Alpamayo** is an end-to-end trajectory planner for autonomous vehicles based on [Alpamayo](https://github.com/NVlabs/alpamayo), NVIDIA's vision-language-action model for autonomous driving. It generates safe driving trajectories by combining:

- **Vision-Language Model (VLM)**: Qwen3-VL-8B processes multi-camera images and produces chain-of-thought (CoT) driving reasoning
- **Expert Diffusion Denoiser**: Flow matching with Euler integration generates trajectory waypoints conditioned on the VLM's understanding
- **TensorRT Expert Acceleration**: The expert denoiser runs as a TRT FP16 engine via ONNX Runtime, compiled and cached at first launch

This node is designed for real-vehicle deployment within the [Autoware](https://autoware.org/) ecosystem, following the same interface as the reference `alpamayo_ros` node from [alpamayo-autoware](https://github.com/NVlabs/alpamayo).

---

## Performance

Benchmarked on **NVIDIA RTX PRO 6000 Blackwell** (96 GB GDDR7, SM120) with 4 cameras × 4 temporal frames = 16 images at 1080×1920 resolution.

| Configuration                                            | End-to-End Latency | FPS     | Trajectory        |
| -------------------------------------------------------- | ------------------ | ------- | ----------------- |
| Baseline (CPU preproc, native expert, sampling, 10-step) | 1031 ms            | 1.0     | Reference         |
| + TRT Expert                                             | 913 ms             | 1.1     | ~3% deviation     |
| + GPU preprocessing                                      | 770 ms             | 1.3     | ~3% deviation     |
| + Greedy decode                                          | 709 ms             | 1.4     | ~5% deviation     |
| **Speed Mode (all optimizations)**                       | **656 ms**         | **1.5** | **~5% deviation** |

> **End-to-end latency** is measured from raw JPEG bytes to trajectory output. All stages included: JPEG decode, image resize, tokenization, VLM generation, expert diffusion, and trajectory decoding. No steps are excluded.

**Speed Mode achieves 36% latency reduction** (1031 → 656 ms) with ~5% trajectory deviation from baseline.

### Enabling Speed Mode (~1.5 FPS)

Set the following parameters in `tensorrt_alpamayo.param.yaml`:

```yaml
num_diffusion_steps: 5 # 5 instead of 10
use_greedy_decode: true # Deterministic decode
```

Speed Mode combines GPU-accelerated preprocessing, TRT FP16 expert engine, greedy decoding, and 5-step diffusion. The trajectory deviates by approximately 5% compared to the default configuration. For safety-critical applications, use the default settings.

---

## Architecture

```text
Camera Topics (CompressedImage × 4)     Odometry Topic
        │                                      │
        ▼                                      ▼
┌─────────────────────────────────────────────────────┐
│           Autoware TensorRT Alpamayo Node            │
│                                                     │
│  Image Decode ──► Tokenizer ──► VLM (PyTorch BF16)  │
│                                      │              │
│                                 KV Cache            │
│                                      │              │
│                          Expert Denoiser (TRT FP16)  │
│                                      │              │
│                          Trajectory Decode           │
└──────────────────────────┬──────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Trajectory    CoT Text     Markers
```

### Key Design Decisions

- **VLM runs in PyTorch BF16** — cuBLAS GEMM with flash attention provides the fastest VLM inference on SM120 (Blackwell consumer GPUs). TensorRT-based VLM inference is slower on SM120 due to kernel fallback.
- **Expert runs in TRT FP16** — The expert denoiser ONNX is compiled to a TensorRT engine at first launch and cached. FP16 with FP32 fallback for LayerNorm/Softmax ensures numerical stability on long KV cache sequences (11k+ tokens).
- **Greedy decode option** — Eliminates top-p/temperature sampling overhead (~60 ms), suitable for deterministic deployment.

---

## How to Use

### Prerequisites

| Requirement | Specification                                        |
| ----------- | ---------------------------------------------------- |
| GPU         | NVIDIA GPU with 24 GB+ VRAM (tested on RTX PRO 6000) |
| CUDA        | 12.8+                                                |
| ROS 2       | Humble                                               |
| Python      | 3.10+ with PyTorch, flash-attn, ONNX Runtime GPU     |

The Alpamayo model weights are downloaded automatically from HuggingFace on first launch. Ensure a valid HuggingFace token is configured:

```bash
huggingface-cli login
```

### Build

```bash
colcon build --packages-select autoware_tensorrt_alpamayo
```

### Launch

```bash
ros2 launch autoware_tensorrt_alpamayo tensorrt_alpamayo.launch.xml
```

### Parameters

| Parameter              | Default                          | Description                                    |
| ---------------------- | -------------------------------- | ---------------------------------------------- |
| `model_id`             | `nvidia/Alpamayo-R1-10B`         | HuggingFace model ID                           |
| `expert_onnx_path`     | `""`                             | Path to expert ONNX (auto-compiles TRT engine) |
| `num_diffusion_steps`  | `10`                             | Diffusion steps (10 = quality, 5 = speed)      |
| `use_greedy_decode`    | `false`                          | Greedy decode for speed                        |
| `top_p`                | `0.98`                           | Nucleus sampling (when not greedy)             |
| `temperature`          | `0.6`                            | Sampling temperature (when not greedy)         |
| `camera_topics`        | See config                       | Camera image topics                            |
| `odometry_topic`       | `/localization/kinematic_state`  | Odometry topic                                 |
| `trajectory_topic`     | `/alpamayo/predicted_trajectory` | Output trajectory                              |
| `inference_period_sec` | `1.0`                            | Inference trigger period                       |

---

## Topics

### Subscriptions

| Topic              | Type                          | Description                 |
| ------------------ | ----------------------------- | --------------------------- |
| `~/input/camera_*` | `sensor_msgs/CompressedImage` | Multi-camera JPEG streams   |
| `~/input/odometry` | `nav_msgs/Odometry`           | Ego vehicle kinematic state |

### Publications

| Topic                         | Type                                | Description                        |
| ----------------------------- | ----------------------------------- | ---------------------------------- |
| `~/output/trajectory`         | `autoware_planning_msgs/Trajectory` | 64-waypoint planned trajectory     |
| `~/output/cot`                | `std_msgs/String`                   | Chain-of-thought driving reasoning |
| `~/output/trajectory_markers` | `visualization_msgs/MarkerArray`    | RViz visualization                 |

---

## References

- [Alpamayo](https://github.com/NVlabs/alpamayo) — NVIDIA's vision-language-action model for autonomous driving (model weights, training, evaluation)
- [alpamayo-autoware](https://github.com/autowarefoundation/alpamayo-autoware) — Reference ROS 2 Python node and Autoware integration by Autoware Foundation
