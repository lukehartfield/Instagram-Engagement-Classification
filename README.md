# 📸 Instagram Engagement Classification using Vision Labels & Captions

> *Can we predict which Instagram posts will go viral based on their image content and captions?*

This project explores the intersection of **computer vision**, **natural language processing**, and **predictive modeling** to classify Instagram posts by their **engagement levels**. By combining **Google Vision API** image labels with **bag-of-words caption analysis**, we built multiple classifiers to determine what content leads to high performance.

---

## 🧠 Project Goals

- Scrape and preprocess Instagram post data
- Generate image labels using **Google Cloud Vision API**
- Compare the predictive power of:
  - Image labels alone  
  - Captions alone  
  - Combined text+vision features
- Evaluate performance using **logistic regression**
- Apply **topic modeling (LDA)** to explore latent themes in high-performing posts

---

## 🗂️ Dataset

| Feature        | Description                                |
|----------------|--------------------------------------------|
| `image_url`    | Scraped directly from brand Instagram feed |
| `caption`      | User-written caption content               |
| `likes`        | Total likes per post                       |
| `image_labels` | Tags generated from Google Vision API      |
| `binary`       | Engagement class: 1 if likes > median, else 0 |

- ~500 total Instagram posts from a fashion-forward brand (e.g., Zara)
- Balanced classes based on **median split** of likes
- Combined into a single CSV for modeling

---

## 🧮 Methods

### 🖼️ Image Label Generation
- Used **Google Cloud Vision API** to extract semantic tags from post images
- Each image returns ~5–15 labels (e.g., "fashion", "sky", "model", "portrait")

### 🧾 Text Preprocessing
- Cleaned captions: lowercasing, punctuation stripping, lemmatization
- Applied `CountVectorizer` to both:
  - Captions
  - Vision labels
- Converted both into **binary feature matrices** (BoW)

### 🔢 Classification Models
- Modeled engagement (`binary`) using **logistic regression**
- 3 input configurations:
  1. Vision Labels Only  
  2. Captions Only  
  3. Combined Vision + Captions
- Evaluated on **train/test split** (80/20) using accuracy and confusion matrix

### 📚 Topic Modeling
- Ran **Latent Dirichlet Allocation (LDA)** on vision label BoW
- Extracted **5 dominant topics**
- Compared topic prevalence in:
  - Top 25% posts by likes  
  - Bottom 25% posts by likes

---

## 📊 Results

### 🔍 Classification Accuracy

| Model                | Accuracy | Notes                                |
|----------------------|----------|--------------------------------------|
| Vision Labels Only   | 74.3%    | Most predictive single feature group |
| Captions Only        | 71.8%    | Some signal, but less than vision    |
| Combined Features    | **78.5%**| Best performing configuration        |

**Confusion Matrix**:
![confusion_matrix](assets/confusion_matrix.png)

- High True Positive Rate: Model successfully identifies most viral posts
- False positives mainly occur in borderline cases with vague content

---

### 📚 LDA Topic Modeling (on Image Labels)

| Topic | Keywords                                 | Description             |
|-------|------------------------------------------|-------------------------|
| 1     | fashion, clothing, sleeve, model         | Apparel-centric posts   |
| 2     | outdoor, sky, leaf, natural              | Environmental content   |
| 3     | portrait, eye, smile, close-up           | Human/face close-ups    |
| 4     | city, urban, modern, trend               | Street/lifestyle shots  |
| 5     | fabric, pattern, contrast, design        | Texture/color emphasis  |

---

### 🎯 Engagement by Topic

| Topic     | Top 25% Posts | Bottom 25% Posts | Interpretation               |
|-----------|----------------|------------------|------------------------------|
| Fashion   | 32%            | 19%              | Strong driver of engagement  |
| Close-up  | 28%            | 12%              | Correlates with high likes   |
| Nature    | 14%            | 30%              | Overused in low-like content |
| Urban     | ~Same          | ~Same            | Neutral impact               |

![topic_dist_plot](assets/topic_engagement.png)

---

## 📌 Key Takeaways

- **Image content > caption text** for engagement prediction
- Vision labels from Google Cloud provide **powerful low-cost features**
- Combining captions + image labels gives the **best performance**
- High-performing posts skew toward:
  - Human close-ups  
  - Fashion-centered framing  
  - Clean textures and subject focus
- NLP-based topic modeling adds explainability to visual prediction models

---

## 🛠️ Tech Stack

| Tool              | Purpose                              |
|-------------------|---------------------------------------|
| `BeautifulSoup`   | Instagram post scraping               |
| `Google Cloud Vision` | Image label extraction              |
| `scikit-learn`    | Logistic regression, BoW modeling     |
| `gensim`          | LDA topic modeling                    |
| `matplotlib`, `seaborn` | Visualizations                   |
| `pandas`, `numpy` | Data cleaning, matrix prep            |

---

## 📁 File Structure

