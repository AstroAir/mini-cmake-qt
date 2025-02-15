**Basic Concept**:
Camera lens calibration is the process of determining the camera's **intrinsic parameters** and **distortion coefficients** to correct optical distortions and establish accurate geometric relationships.

**Key Parameters to Calibrate**:

1. **Intrinsic Parameters**:
   $$\begin{bmatrix}
   f_x & 0 & c_x \\
   0 & f_y & c_y \\
   0 & 0 & 1
   \end{bmatrix}$$
   Where:
   - $f_x, f_y$ = **focal lengths** in pixels
   - $c_x, c_y$ = **principal point** coordinates

2. **Distortion Coefficients**:
   - **Radial Distortion**: $k_1, k_2, k_3$
   - **Tangential Distortion**: $p_1, p_2$

**Calibration Process**:

1. **Data Collection**:
   - Use calibration pattern (typically checkerboard)
   - Capture 10-20 images from different angles
   - Ensure pattern covers different areas of image

2. **Pattern Detection**:
   - Find corner points
   - Extract 2D-3D point correspondences

3. **Parameter Estimation**:
   - Minimize reprojection error
   - Solve optimization problem

**Implementation with OpenCV**:

```cpp
// Basic structure for calibration
std::vector<std::vector<cv::Point3f>> objectPoints;
std::vector<std::vector<cv::Point2f>> imagePoints;
cv::Mat cameraMatrix, distCoeffs;
std::vector<cv::Mat> rvecs, tvecs;

// Calibration function
cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                    cameraMatrix, distCoeffs, rvecs, tvecs);
```

**Quality Metrics**:

1. **Reprojection Error**:
   $$E = \sum_{i=1}^n \|\text{projected}(P_i) - \text{measured}(p_i)\|^2$$

2. **RMS Error**:
   $$RMS = \sqrt{\frac{\sum_{i=1}^n (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}{n}}$$

**Applications**:

1. **Computer Vision**:
   - 3D reconstruction
   - Visual SLAM
   - Object detection

2. **Industrial**:
   - Quality inspection
   - Robotic vision
   - Measurement systems

**Common Issues and Solutions**:

1. **Pattern Quality**:
   - **Issue**: Poor pattern detection
   - **Solution**: Use high-contrast patterns

2. **Sample Distribution**:
   - **Issue**: Insufficient views
   - **Solution**: Capture images at various angles

3. **Lighting Conditions**:
   - **Issue**: Inconsistent lighting
   - **Solution**: Ensure uniform illumination

**Best Practices**:

1. **Pattern Selection**:
   - **Checkerboard**: Most common
   - **Size**: 9x6 or 8x6 squares
   - **Square size**: Known physical dimension

2. **Image Capture**:
   - **Coverage**: Full field of view
   - **Angles**: 20-30 degrees variation
   - **Distance**: Various distances

3. **Validation**:
   - **Cross-validation**
   - **Test on independent datasets**
   - **Regular recalibration**

**Advanced Techniques**:

1. **Multi-camera Calibration**:
   - Stereo calibration
   - Array calibration

2. **Dynamic Calibration**:
   - Online calibration
   - Self-calibration

3. **Specialized Calibration**:
   - Fisheye cameras
   - Thermal cameras
   - Wide-angle lenses

**Error Sources**:

1. **Mechanical**:
   - Lens mounting
   - Sensor alignment

2. **Optical**:
   - Lens distortion
   - Chromatic aberration

3. **Digital**:
   - Sensor noise
   - Discretization errors

**Documentation Requirements**:

- Calibration date
- Environmental conditions
- Equipment used
- Calibration results
- Validation metrics

This comprehensive approach ensures accurate camera calibration, which is crucial for many computer vision applications.
