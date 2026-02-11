# Lunar Mission Planning System

A computer vision-based navigation system for autonomous lunar surface operations using Python and OpenCV.

## 🚀 Quick Start

### Requirements
- Python 3.7+
- Webcam (or Raspberry Pi Camera)
- Paper with drawn lines/paths for testing

### Installation

1. Install dependencies:
```bash
pip install opencv-python numpy networkx matplotlib scipy
```

For Raspberry Pi Camera (optional):
```bash
pip install picamera2
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## 📚 Step-by-Step Testing

Run each step to build and test the system incrementally:

### Step 1: Camera Test
```bash
python step1_camera_test.py
```
**What it does:** Opens your webcam and displays live feed  
**Success:** You see yourself on screen  
**Controls:** Press 'q' to quit, 's' to save a test image

### Step 2: Preprocessing
```bash
python step2_preprocessing.py
```
**What it does:** Converts to grayscale and applies blur  
**Success:** See 3 views: Original | Grayscale | Blurred  
**Controls:** Press 'q' to quit

### Step 3: Edge Detection
```bash
python step3_edge_detection.py
```
**What it does:** Finds edges using Canny algorithm  
**Success:** See edges of objects highlighted  
**Controls:** Press 'q' to quit, '+/-' to adjust sensitivity

### Step 4: Line Detection
```bash
python step4_line_detection.py
```
**What it does:** Detects lines using Hough Transform  
**Success:** Green lines drawn over detected paths  
**Tip:** Point camera at paper with drawn lines  
**Controls:** Press 'q' to quit, 't' to adjust threshold

### Step 5: Obstacle Detection
```bash
python step5_obstacle_detection.py
```
**What it does:** Identifies obstacles using contours  
**Success:** Red boxes around detected obstacles  
**Controls:** Press 'q' to quit, 'a' to adjust sensitivity

### Step 6: Path Planning (A*)
```bash
python step6_path_planning.py
```
**What it does:** Computes optimal path avoiding obstacles  
**Success:** Green path drawn from start to goal  
**How to use:**
1. Click once to set START point (green circle)
2. Click again to set GOAL point (red circle)
3. Watch it compute the path!
**Controls:** Press 'q' to quit, 'r' to reset points

## 🎯 Testing Tips

### Best Test Setup:
1. **For Line Detection (Step 4):**
   - Draw thick black lines on white paper
   - Use marker, not pencil
   - Create intersections and paths

2. **For Obstacle Detection (Step 5):**
   - Place dark objects on light surface
   - Use books, boxes, cups as "obstacles"

3. **For Path Planning (Step 6):**
   - Combine lines + obstacles
   - Draw paths with obstacles in between
   - Click start/goal on opposite sides

### Common Issues:

**"Cannot open webcam"**
- Change `CAMERA_INDEX` in config.py (try 0, 1, 2)
- Check if another program is using camera

**"No lines detected"**
- Increase contrast (darker lines on lighter background)
- Press 't' to adjust threshold
- Make lines thicker

**"No path found"**
- Start/goal might be in obstacle
- Try different points
- Reduce obstacle safety margin in code

## 📁 Project Structure

```
lunar_mission_planner/
├── step1_camera_test.py       # Camera initialization
├── step2_preprocessing.py     # Image preprocessing
├── step3_edge_detection.py    # Canny edge detection
├── step4_line_detection.py    # Hough line detection
├── step5_obstacle_detection.py # Obstacle detection
├── step6_path_planning.py     # A* pathfinding
├── config.py                  # Configuration settings
├── logger.py                  # Logging utilities
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

Edit `config.py` to adjust:
- Camera resolution and FPS
- Edge detection sensitivity
- Line detection parameters
- Obstacle detection thresholds
- Path planning settings

## 🎓 How It Works

### 1. Image Acquisition
Camera captures live feed at configured resolution

### 2. Preprocessing
- Convert to grayscale
- Apply Gaussian blur to reduce noise

### 3. Feature Detection
- **Lines:** Canny edge detection + Hough Transform
- **Obstacles:** Adaptive thresholding + Contour detection

### 4. Path Planning
- Build navigation graph from detected features
- Use A* algorithm with Euclidean distance heuristic
- Avoid obstacles with safety margin

### 5. Visualization
- Green lines = detected paths
- Red boxes = obstacles
- Green circle = start point
- Red circle = goal point
- Green path = computed route

## 🚀 Next Steps

After testing all steps individually:
1. Combine all modules into single application
2. Add base station GUI
3. Implement wireless communication
4. Add Pi Camera support for actual CubeSat deployment

## 📝 Academic Notes

**Algorithms Used:**
- Canny Edge Detection
- Probabilistic Hough Transform
- Adaptive Thresholding
- Morphological Operations
- A* Search Algorithm

**Applications:**
- Autonomous robot navigation
- Mars rover pathfinding
- Lunar surface exploration
- General computer vision

## 🐛 Troubleshooting

**Low FPS:**
- Reduce camera resolution in config.py
- Close other programs

**Path calculation slow:**
- Reduce image resolution
- Increase step_size in A* function

**False obstacle detection:**
- Adjust lighting
- Increase min_area threshold
- Use uniform background

## 📄 License

Educational project for BWSI CubeSat program.

## 🤝 Contributing

This is a learning project. Feel free to experiment and modify!

---

**Ready to test?** Start with Step 1!

```bash
python step1_camera_test.py
```
