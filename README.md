# Handwritten Text Recognition (HTR) for Bosnian Printed Letters

The project is based on code from the [handwritten-text-recognition](https://github.com/arthurflor23/handwritten-text-recognition) and [master-theses-HTR](https://github.com/abegovac2/masters-theses-HTR) repositories.

This project presents a system for recognizing **handwritten printed text in the Bosnian language**, developed as part of the Artificial Intelligence course at the University of Sarajevo.

It utilizes the **HTR-FLORe neural network model**, combining convolutional and recurrent layers trained on a custom dataset. The goal is to build a robust handwritten text recognition model that operates effectively on a low-resource language like Bosnian.

---

## Project Overview

- **Language:** Bosnian
- **Text type:** Printed handwritten text (štampana slova)
- **Model:** FLORe (Full-Gated Convolutional Recurrent Network)
- **Evaluation metrics:** CER, WER, SER
- **Tools:** TensorFlow 2.x, Python, Google Colab, HDF5 format

---

## Workflow

1. **Dataset Creation**
   - Texts sourced from literature, the web, and generated content.
   - Handwritten in 3 styles by 8 authors and scanned at 300 DPI.
   - Annotated using LabelImg in PascalVOC format.
   - Focused on printed letters (štampana slova).
   - Final dataset:  
     - 11,592 words  
     - 1,359 lines (after segmentation correction)

2. **Preprocessing**
   - Extraction of individual words and lines (two methods).
   - Segmentations converted to `.txt` and cropped into `.jpg` images.
   - Data split: 70% train, 20% validation, 10% test.
   - Converted to `.hdf5` format for training.

3. **Model Training**
   - FLORe architecture with CTC loss and bi-directional GRU layers.
   - Trained on:
     - Individual words (`labels_w`)
     - Lines (original segmentation) (`labels_l`)
     - Lines (modified segmentation) (`labels_l_m`)
   - Early stopping used to prevent overfitting.

4. **Evaluation**
   - Metrics:  
     - **CER (Character Error Rate)**  
     - **WER (Word Error Rate)**  
     - **SER (Sequence Error Rate)**
   - Comparison with LLMs (GPT-4o, Claude, Gemini) included.

---

## Results Summary

| Dataset       | CER     | WER     | SER     |
|---------------|---------|---------|---------|
| Words         | **Low** | **Low** | **Low** |
| Lines (orig.) | Medium  | Medium  | **Lower** than modified |
| Lines (mod.)  | Slightly better CER, slightly worse WER & SER |

- Best overall performance achieved with individual word dataset.
- Modified segmentation method was structurally better but yielded slightly worse metrics due to longer sequences.
- LLMs showed potential but were still limited for Bosnian.

---

## Project Structure

![image](https://github.com/user-attachments/assets/a4c74e12-a5a5-48f4-8fc5-7f8fce7a6027)

---

## Performance Analysis

- Early Stopping was triggered in all trainings (best at 106–135 epochs).
- No overfitting observed when training on words.
- Overfitting occurred with line-based datasets due to longer input sequences.
- Modified segmentation improved dataset structure but introduced metric trade-offs.

---

## Comparison with LLMs

- LLMs (GPT-4o, Claude, Google Gemini) were tested on the same handwritten samples.
- Results were promising but notably weaker for Bosnian than for English.
- Traditional HTR still outperforms LLMs on handwritten recognition tasks, especially for non-English texts.

---

## Demo

- ### **Screenshots of input images with model predictions**

 ### **Words** <br>
<img src="https://github.com/user-attachments/assets/4b1c9692-4f7a-4d79-a376-e8a886f78d6a" width="80%"/>

 ### **Lines** (orig.) <br>
 <img src="https://github.com/user-attachments/assets/b7fff16f-c881-4cc7-abde-657b5ddf7e5a" width="80%"/>

 ### **Lines** (mod.) <br>
 <img src="https://github.com/user-attachments/assets/05bcfe1c-3ae9-4b5b-b618-4c3f95f02ed3" width="80%"/>

- ### **Training and validation loss plots**  
 ### **Words** <br>
<img src="https://github.com/user-attachments/assets/f599469d-080e-4664-b504-89f31e1a1108" width="50%"/>

### **Lines** (orig.) <br>
<img src="https://github.com/user-attachments/assets/ae2ff275-0e16-474c-8b3d-35e79b076cf5" width="50%"/>

### **Lines** (mod.) <br>
<img src="https://github.com/user-attachments/assets/c5de00f5-9d46-4deb-afa3-bdc1a629fc21" width="50%"/>


- ### **Evaluation metrics (CER, WER, SER)**  
![image](https://github.com/user-attachments/assets/c61eabc9-9637-4b29-a950-ebd380efbb31)

- ### **Google Colab Notebook**  
A full training and evaluation pipeline is available in the repository:  
📎 [`HTR_bos_stampana_slova.ipynb`](https://github.com/ijozic1/Handwritten-text-recognition/blob/main/HTR_bos_stampana_slova.ipynb)

---

## Authors
- [Ivona Jozić](https://github.com/ijozic1)
- [Ismar Muslić](https://github.com/imuslic1)
- Supervised by: Prof. dr Amila Akagić <br>
University of Sarajevo, Faculty of Electrical Engineering

---
