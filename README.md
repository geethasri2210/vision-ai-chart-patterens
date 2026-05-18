# vision-ai-chart-patterens
AI model to detect candlestick chart patterns using CNNs in TensorFlow
# Vision AI Chart Pattern Recognition

AI model to detect candlestick chart patterns using Convolutional Neural Networks (CNNs) in TensorFlow.


##  Overview
This project applies deep learning to financial candlestick charts, enabling automatic detection of chart patterns such as Bullish Engulfing, Doji, Hammer, and others.  
The goal is to assist traders and analysts by providing fast, reliable pattern recognition.

---

##  Project Structure
- `data/` → Raw candlestick chart images
- `data-preprocessing/` → Scripts for cleaning and preparing data
- `train.py` → Model training script
- `model.py` → CNN architecture definition
- `demo.py` → Run predictions on sample charts
- `README.md` → Project documentation

---

## ⚙️ Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/geethasri2210/vision-ai-chart-patterens.git
   cd vision-ai-chart-patterens


2. Install dependencies
   pip install -r requirements.txt
3. Prepare datasets:
    Place candlestick images in data/.
    Run preprocessing scriprts in data preprocessing/.
## Training the Model
 run:
 python train.py
This will tarin the CNN on candlestick chart images and save the model weights.
## Demo Prediction
  Run:
  python demo.py --image sample_chart.png
  Output:
   Predicted pattern: Double Bottom

## RESULT
  Achieved ~85% accuracy on test dataset
  successfully detects common candlestick pattrens

## FUTURE SCOPE
   Integrate with live trading API's for real time Predictions




## AUTHOR

 Geetha Sri Mopidevi
 B.Tech Student | Aspiring AI Engineer
 LinkedIn(https://www.linkedin.com/in/geetha-sri-a28266257/)
 Github(https://github.com/geethasri2210)








