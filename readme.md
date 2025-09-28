# Interactive Image Mosaic Generator

A Python implementation for creating high-quality image mosaics using advanced tile-matching algorithms. This project combines computer vision techniques with efficient caching to generate visually appealing mosaics that reconstruct input images using collections of smaller tile images.

## Features

### Core Functionality
- **Dual Matching Algorithms**: Color-only matching and hybrid color+texture matching
- **Multi-Scale Processing**: Support for tile sizes from 4×4 to 32×32 pixels
- **Comprehensive Tile Preprocessing**: Automatic rotation (0°, 90°, 180°, 270°) and flipping variants
- **Smart Caching System**: Efficient preprocessing with automatic cache invalidation
- **Quality Assessment**: Multiple evaluation metrics including SSIM, PSNR, and MSE

### Advanced Features
- **LAB Color Space**: Perceptually uniform color matching for better visual results
- **Texture Analysis**: Sobel gradient-based texture features for enhanced matching
- **Color Quantization**: Optional palette reduction for artistic effects
- **Grid Visualization**: Overlay grid lines for analysis and debugging
- **Batch Processing**: Support for multiple grid sizes and configurations

### User Interface
- **Gradio Web Interface**: Interactive parameter adjustment with real-time preview
- **Sample Images**: Built-in examples for testing and demonstration
- **Download Functionality**: Save generated mosaics directly from the interface
- **Performance Metrics**: Detailed timing and quality analysis

## Project Structure

```
mosaic_generator/
├── app.py                 # Main Gradio application interface
├── mosaic_core.py         # Core image processing algorithms
├── tile_cache.py          # Tile preprocessing and caching system
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── samples/               # Sample input images
│   ├── beach.jpg
│   ├── husky_puppy.jpg
│   └── rdj.jpg
├── Tiles/                 # Tile image collection
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── cache/                 # Auto-generated cache files
    └── *.npz
```

## Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare tile collection**:
   - Add tile images to the `Tiles/` directory
   - Supported formats: JPG, PNG, BMP, WebP
   - Recommended: 200-1000 diverse, high-quality images

## Usage

### Web Interface (Recommended)

Launch the interactive web application:

```bash
python app.py
```

The interface will be available at `http://localhost:7860`

#### Interface Controls:
- **Input Image**: Upload your image or select from samples
- **Tile Directory**: Path to your tile collection (default: "Tiles")
- **Grid Size**: Output dimensions (16×16 to 128×128)
- **Tile Size**: Individual tile dimensions (4×4 to 32×32 pixels)
- **Method**: Choose between "Only Color based" or "Custom (Color+Texture)"
- **Quantization**: Optional color reduction (0=disabled, 8-32 colors)
- **Alpha (α)**: Color vs texture weight for custom method (0.0-1.0)

### Algorithm Details

#### Color-Only Matching
Uses z-score normalized Euclidean distance in LAB color space:
- Converts RGB to perceptually uniform LAB color space
- Computes mean colors for image blocks and tiles
- Normalizes features using z-score standardization
- Finds optimal matches using efficient distance computation

#### Custom (Color+Texture) Matching
Hybrid approach combining color and texture similarity:
- **Color component**: LAB color space mean matching
- **Texture component**: Sobel gradient magnitude analysis
- **Weighting**: Alpha parameter balances color vs texture importance
- **Normalization**: Distances scaled to [0,1] for fair combination

### Quality Metrics

The system provides comprehensive quality assessment:

- **MSE**: Mean Squared Error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio in dB (higher is better)
- **SSIM (pixel)**: Global structural similarity (higher is better)
- **SSIM (block)**: Per-tile structural similarity
- **SSIM (R,G,B)**: Per-channel analysis
- **SSIM (global gray)**: Grayscale structural similarity
- **SSIM (multi-scale gray)**: Multi-resolution analysis

#### Quality Interpretation:
- **SSIM > 0.85**: Excellent quality
- **SSIM > 0.75**: Good quality
- **SSIM > 0.65**: Fair quality
- **SSIM < 0.65**: Poor quality, consider parameter adjustment

## Performance Optimization

### Tile Collection Guidelines
- **Count**: 500-2000 tiles provide good variety
- **Resolution**: Source tiles should be larger than target tile size
- **Diversity**: Include varied colors, textures, and patterns
- **Quality**: Use high-resolution, well-lit images

### Parameter Tuning
- **Alpha = 1.0**: Pure color matching (fastest)
- **Alpha = 0.7**: Balanced approach (recommended)
- **Alpha = 0.3**: Texture-focused (best for detailed images)
- **Grid Size**: Larger grids provide more detail but increase processing time
- **Tile Size**: Smaller tiles capture finer details but may reduce coherence

### Processing Times
- **Cache Building**: One-time cost of 1 minute approx for all tile sizes
- **Single Mosaic**: 10-70 seconds approx depending on grid size and tile size
- **Memory Usage**: 200-500MB for typical configurations

## Technical Implementation

### Core Modules

#### `mosaic_core.py`
- Image I/O with error handling
- Grid-based resizing and cropping
- Feature extraction (LAB means, texture measures)
- Mosaic reconstruction algorithms
- Comprehensive quality metrics

#### `tile_cache.py`
- Tile preprocessing with transformations
- Compressed cache storage (.npz format)
- Automatic cache invalidation
- Batch processing utilities

#### `app.py`
- Gradio web interface
- Parameter validation and error handling
- Real-time performance monitoring
- File download functionality

### Dependencies

```
numpy>=1.24            # Numerical computing
Pillow>=9.5            # Image processing
scikit-image>=0.22     # Computer vision algorithms
scipy>=1.10            # dependency for scikit-image
gradio==5.47.1         # Web interface
```

## File Format Support

- **Input Images**: JPG, PNG, BMP, WebP
- **Tile Images**: JPG, PNG, BMP, WebP
- **Output**: PNG format with lossless compression
- **Cache**: NumPy compressed format (.npz)

## Cache Management

The system automatically manages tile caches:
- **Automatic Building**: Caches built on first use
- **Invalidation**: Detects changes in tile directory
- **Compression**: Efficient storage using NumPy compression
- **Versioning**: Feature version tracking for compatibility

Cache files are stored in the `cache/` directory with naming format:
`tiles_s{SIZE}_allrotflips_{DIGEST}.npz`

## Error Handling

The application includes comprehensive error handling for:
- Missing or corrupted input images
- Invalid tile directories
- Insufficient system resources
- Cache corruption or incompatibility
- Network issues (for web interface)

## Troubleshooting

### Common Issues:

1. **"No images found in Tiles"**
   - Verify tile directory path
   - Check file formats (JPG, PNG, BMP, WebP)
   - Ensure directory contains valid images

2. **Cache building fails**
   - Check disk space (cache requires ~50-200MB per tile size)
   - Verify read permissions for tile directory
   - Remove corrupted cache files and rebuild

3. **Poor mosaic quality**
   - Increase tile collection diversity
   - Adjust alpha parameter for better color/texture balance
   - Try different quantization settings
   - Use smaller tile sizes for finer detail

4. **Slow processing**
   - Reduce grid size for faster processing
   - Use larger tile sizes
   - Ensure adequate system RAM (500MB+ recommended)

### Performance Tips:
- Keep tile collections under 2000 images for optimal performance
- Use SSD storage for cache directory to improve I/O speed
- Close other applications to free system memory
- Consider using smaller input images for testing parameters

## License

This project is developed for educational and research purposes. Feel free to use and modify for non-commercial applications.

## Contributing

Contributions welcome for:
- Additional matching algorithms
- Performance optimizations
- UI/UX improvements
- Documentation enhancements
- Bug fixes and testing

---

**Create stunning mosaic art with advanced computer vision algorithms!**