# ♻️ Waste Detection AI (YOLOv8 + Streamlit)

An AI-powered waste detection system that identifies waste types such as **plastic, paper, glass, metal, etc.** using a custom-trained **YOLOv8 object detection model**.
The application includes a **real-time webcam detection feature** and a **user-friendly Streamlit interface**.

---

## 🚀 Features

* Real-time waste detection using webcam
* Image upload detection
* Custom trained YOLOv8 model
* Malayalam-ready UI support (extendable)
* Bounding box + confidence score visualization
* Lightweight and fast inference

---

## 🧠 Model Details

* Model: **YOLOv8n**
* Framework: PyTorch + Ultralytics
* Training Dataset: Custom waste dataset (Roboflow format)
* Classes:

  * Plastic
  * Paper
  * Glass
  * Metal
  * Cardboard
  * Organic (extendable)

---


## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/divyaeldho/Waste-detection-ai.git
cd Waste-detection-ai
```

Create environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run App

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## 📂 Project Structure

```
Waste-detection-ai
│── train/
│── valid/
│── test/
│── app.py
│── data.yaml
│── requirements.txt
│── .gitignore
```

---

## 📊 Future Improvements

* Location-based waste logging
* Daily report generation
* Excel export reports
* Municipal monitoring dashboard
* Mobile deployment

---



If you like this project, consider giving it a ⭐ on GitHub!
