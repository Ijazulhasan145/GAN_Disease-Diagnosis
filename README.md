# Disease Diagnosis Prediction using GANs

This project implements a **Generative Adversarial Network (GAN)** to generate synthetic healthcare data for disease diagnosis prediction. GANs are a powerful deep learning tool that can learn to create data distributions similar to real datasets — particularly useful when real-world data is scarce or sensitive, as is often the case in the healthcare domain.

## Objective

To simulate realistic patient data that includes clinical features (e.g., body temperature, blood pressure, symptoms duration) and use it to model disease diagnosis prediction, ensuring data privacy and improving ML training pipelines.

---

## Features

- **Synthetic Data Generation** using NumPy & Pandas
- **GAN Architecture** implemented with PyTorch
  - Custom Generator & Discriminator models
- **Data Normalization** using MinMaxScaler
- **Training Loop** with Binary Crossentropy Loss
- **Evaluation Metrics**:
  - Generator & Discriminator Loss Curves
  - Feature-wise distribution comparison
- **Visualization** using Matplotlib & Seaborn

---

## Dataset (Synthetic)

The generated dataset includes:
- Age  
- Gender  
- Symptoms Duration  
- Body Temperature  
- Heart Rate  
- Blood Pressure  
- Chronic Illness (Yes/No)  
- Smoking Habit (Yes/No)  
- Water Intake (liters)  
- Diagnosis (0 = No Disease, 1 = Disease)

---

## GAN Model Architecture

### Generator
- Input: Random noise vector
- Layers: Fully Connected (Linear), ReLU, Sigmoid
- Output: Synthetic Patient Record

### Discriminator
- Input: Real or Fake Data
- Layers: Fully Connected (Linear), LeakyReLU, Sigmoid
- Output: Probability of authenticity

---

## Results

- Successfully generated realistic-looking healthcare records.
- Visual analysis showed close distribution match between real and generated data.
- Potential to expand and simulate real-world health scenarios for ML tasks.

---

## Tech Stack

- Python  
- PyTorch  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/gan-disease-diagnosis.git
cd gan-disease-diagnosis

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Disease_Diagnosis_GAN.ipynb
