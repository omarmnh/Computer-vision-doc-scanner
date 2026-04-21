# Computer-vision-doc-scanner
This project focuses on building a computer vision-based document scanner using Python and OpenCV. The goal is to transform a photo of a document into a clean, flat, and readable scanned version.
# 📄 Document Scanner Project

## 📌 Description
This project focuses on building a computer vision-based document scanner using Python and OpenCV. The goal is to transform a photo of a document into a clean, flat, and readable scanned version.

## 🎯 Objectives

* Detect a document in an image
* Extract its 4 corners
* Apply perspective transformation
* Enhance the image for a clean scanned effect
* Provide a simple user interface

---

## 🧠 Technologies Used

* Python
* OpenCV
* NumPy
* Streamlit

---

## ⚙️ How It Works

### 1. Image Preprocessing

* Resize image
* Convert to grayscale
* Apply Gaussian blur

### 2. Edge Detection

* Use **Sobel filter** to detect edges

### 3. Contour Detection

* Detect contours in the image
* Sort them by area
* Select the largest contour

### 4. Document Detection

* Approximate contour shape
* Keep the contour with **4 corners**

### 5. Perspective Transform

* Warp the image to get a top-down view

### 6. Enhancement

* Convert to black & white
* Apply adaptive thresholding

---
<img width="1408" height="768" alt="pipeline document scanner" src="https://github.com/user-attachments/assets/42767517-cac2-4787-b287-a399f8170f06" />


## 🖥️ User Interface

A simple interface built with Streamlit allows users to:

* Upload an image
* Scan the document
* View results (original, warped, enhanced)
* Download the final scanned document

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/document-scanner.git
cd document-scanner
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```bash
document-scanner/
│── app.py              # Streamlit interface
│── scanner.py         # Document scanning algorithm
│── requirements.txt
│── README.md
│── images/            # Test images (optional)
```

---

## 📸 Results

* Original image
* Warped (flattened) document
* Enhanced scanned output

(Add screenshots here if possible)

---

## 💡 Challenges Faced

* Detecting the correct contour
* Handling lighting conditions
* Improving scan quality

---

## ✅ Conclusion

This project demonstrates how fundamental computer vision techniques can be combined to create a practical and useful application.

---


