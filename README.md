# Handwritten Text Recognition (HTR) for Bosnian Printed Letters

The project is based on code from the [handwritten-text-recognition](https://github.com/arthurflor23/handwritten-text-recognition) repository.

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

## Authors
- Ivona Jozić
- Ismar Muslić
- Supervised by: Prof. dr Amila Akagić <br>
University of Sarajevo, Faculty of Electrical Engineering

---
