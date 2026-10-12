# AI-Based Railway Track Defect Detection

A web application for detecting defects in railway tracks using AI and deep learning.

## Features

- 🖼️ **Image Upload**: Upload railway track images via drag-and-drop or file selection
- 🔍 **AI-Powered Analysis**: Deep learning model detects defects with confidence scores
- 📊 **Real-time Results**: Instant feedback with probability percentages
- 🚂 **Defect Classification**: Identifies potential defect types (cracks, surface defects, alignment issues)
- 🎨 **Modern UI**: Clean, responsive interface with gradient design

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or download this project**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Create necessary directories**
   ```bash
   mkdir uploads
   mkdir models
   ```

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to: `http://localhost:5000`

3. **Upload an image**
   - Click the upload area or drag and drop an image
   - Supported formats: PNG, JPG, JPEG, GIF
   - Maximum file size: 16MB

4. **Analyze the image**
   - Click "Analyze Image" button
   - View the results with defect probability and confidence scores

## Training Your Own Model

### Data Preparation

Organize your training data in the following structure:

```
data/
├── train/
│   ├── defect/      # Images of defective tracks
│   └── normal/      # Images of normal tracks
└── validation/
    ├── defect/
    └── normal/
```

### Training the Model

Run the training script:

```bash
python train_model.py
```

The trained model will be saved as `models/defect_detection_model.h5`

### Model Architecture

The current model uses a CNN architecture with:
- 4 Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout for regularization
- Softmax output for binary classification

## Project Structure

```
.
├── app.py                    # Flask application with API endpoints
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── templates/
│   └── index.html           # Frontend interface
├── uploads/                 # Directory for uploaded images (auto-created)
└── models/                  # Directory for trained models (auto-created)
```

## API Endpoints

### `GET /`
- Returns the main web interface

### `POST /upload`
- Accepts image file for analysis
- Returns JSON with defect detection results
- Response format:
  ```json
  {
    "success": true,
    "is_defective": true,
    "defect_probability": 75.5,
    "normal_probability": 24.5,
    "confidence": 75.5,
    "message": "Defect detected!",
    "defect_types": ["Crack", "Surface defect"]
  }
  ```

### `GET /health`
- Health check endpoint
- Returns model loading status

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript (no external frameworks)
- **AI/ML**: TensorFlow/Keras
- **Image Processing**: Pillow (PIL)

## Notes

- The application currently uses a placeholder inference system if no trained model is found
- For production use, train the model with your own dataset of railway track images
- The model expects input images of size 224x224 pixels
- Consider using a pre-trained model (e.g., MobileNet, ResNet) for better performance

## Future Enhancements

- [ ] Add video analysis support
- [ ] Implement batch processing for multiple images
- [ ] Add user authentication
- [ ] Create a database to store analysis history
- [ ] Add PDF report generation
- [ ] Implement API key authentication
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, please refer to the project documentation or contact the development team.
