# Ball-Tracking

A Python-based project for tracking rugby balls using the Ultralytics YOLO model. This project includes four scripts designed for live ball tracking, video processing, and image extraction. The system leverages trained YOLO models (`best.pt` and `rugby.pt`) for accurate detection and tracking.

## Features
- **Live Video Tracking**: Real-time ball detection with play direction overlay.
- **Reduced Frame Tracking**: Optimized for frame sampling to improve performance.
- **Video to Image Conversion**: Extract frames from a video and save as images.
- **Video Ball Tracking**: Annotates video frames with ball tracking and play direction.

## Scripts Overview

### `main.py`
- Tracks the rugby ball in live video streams using `rugby.pt`.
- Displays the direction of play (`left`, `center`, `right`) on the screen.
- Features real-time YOLO model inference and annotations.

### `reduceframes.py`
- Processes a video by sampling frames for efficient ball tracking using `best.pt`.
- Annotates frames with a bounding box and play direction.
- Designed for performance optimization by reducing the number of processed frames.

### `toimage.py`
- Converts video frames into individual images at a specified interval.
- Saves images to a specified output directory for further analysis.

### `vid.py`
- Tracks the rugby ball in video files using `rugby.pt`.
- Annotates the video with play direction (`left`, `center`, `right`) and bounding boxes.
- Implements multi-threaded processing for improved performance.

## Usage

### Live Video Tracking
Run the `main.py` script to track the ball in a live video feed:
```bash
python main.py
```

### Video Frame Sampling and Ball Tracking
To track the ball with reduced frames:
```bash
python reduceframes.py
```

### Convert Video to Images
Extract video frames as images:
```bash
python toimage.py
```

### Process Video Files with Ball Tracking
Run the `vid.py` script for video ball tracking:
```bash
python vid.py
```

## Models
Ensure the YOLO models (`best.pt` and `rugby.pt`) are trained for rugby ball detection and are accessible in the project directory.

## License

This project is licensed under the MIT License.

---

### MIT License
```
MIT License

Copyright (c) 2024 Ralph King

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
