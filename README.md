# ImageNet: A Large-Scale Hierarchical Image Database

The paper introduces ImageNet, a large-scale image database built upon the hierarchical structure of WordNet. ImageNet aims to populate the majority of WordNet’s 80,000 synsets with 500–1000 high-quality, full-resolution images per synset, creating a dataset with tens of millions of annotated images organized semantically.

## 1. Scale & Structure

### ImageNet’s Hierarchical Design:
- **Foundation:** Built on **WordNet**, which organizes words into *synonym sets* (synsets) connected by semantic relationships (e.g., *IS-A* hierarchy: “dog” *is a* “mammal”).
- **Scope:** Aims to cover **80,000 noun synsets** with **500–1000 images per synset**, targeting **tens of millions of images**.
- **Current (2009) Status:**
  - **3.2 million images** across **5247 synsets**.
  - Organized into 12 subtrees: *mammals*, *birds*, *vehicles*, *flowers*, *fruits*, *musical instruments*, *tools*, etc.

### Hierarchy Example:
```
Mammal → Carnivore → Canine → Dog → Working Dog → Husky
```
This structure allows models to generalize from broad categories (e.g., mammal) to specific ones (e.g., Siberian Husky).

### Why It’s Unique:
- **Density:** Unlike prior datasets, ImageNet provides deep hierarchies. For instance, it contains images for **147 dog breeds**, far surpassing datasets like Caltech-256.

## 2. Data Collection Process

### Step 1: Gathering Candidate Images
- **Sources:** Images were scraped from the internet using search engines like Google, Yahoo, and Flickr.
- **Query Strategy:** For each synset, multiple queries were generated using synonyms and translated into languages like Chinese, Spanish, and Italian to diversify results.
- **Volume:** Collected over **10,000 candidate images per synset** before filtering.

### Step 2: Cleaning & Verification (Using Amazon Mechanical Turk - AMT)
- **Human Annotation:** Workers reviewed batches of images, verifying whether each image matched the synset’s definition.
- **Quality Control:**
  - **Redundancy:** Each image was labeled by multiple people to ensure consensus.
  - **Dynamic Thresholding:** More complex categories (e.g., distinguishing between “Burmese cat” and “Siamese cat”) required higher agreement thresholds.

### Precision:
Achieved **~99.7% labeling accuracy**, especially impressive given the dataset’s scale.

## 3. Comparison to Other Datasets

| **Dataset**       | **Categories** | **Images**       | **Resolution** | **Hierarchy** |
|-------------------|---------------|------------------|----------------|---------------|
| **ImageNet**      | 5247+         | 3.2 million      | Full-res       | Dense         |
| Caltech-256       | 256           | 30,607           | Medium         | Shallow       |
| TinyImages        | 80 million    | Low-quality      | 32×32 pixels   | None          |
| ESP Game         | ~60,000       | Moderate         | Varies         | Basic labels  |
| LabelMe          | 30,000        | Annotated        | High-res       | Limited       |

### What Makes ImageNet Stand Out:
- **Scale:** Orders of magnitude larger in terms of both categories and total images.
- **Accuracy:** High precision labeling thanks to AMT verification.
- **Hierarchy:** Images are semantically organized, enabling tasks like fine-grained classification (e.g., distinguishing 200+ bird species).

## 4. Applications of ImageNet

### a. Object Recognition & Classification

#### Baseline Methods Tested:
1. **Nearest Neighbour (NN) Voting:** Classifies objects by comparing them to the closest images in the dataset.
2. **Naive Bayes Nearest Neighbour (NBNN):** Uses feature descriptors (like SIFT) for more sophisticated classification.

#### Findings:
- Models trained on **clean ImageNet data** outperformed those using noisy datasets (like TinyImages).
- **High-resolution images** improved performance significantly.

### b. Tree-Based Image Classification

- **Tree-Max Classifier:**  
  Instead of treating each category separately, this method **exploits the hierarchical structure**. For example:
  - If a model detects “Golden Retriever,” it automatically considers broader categories like “Dog” and “Mammal.”

- **Results:**  
  - Classification performance improved at every tree level compared to flat classifiers.
  - **Stronger performance for fine-grained categories** (e.g., “minivan” vs. just “vehicle”).

### c. Object Localization & Clustering

- **Automatic Localization:**  
  Applied models to **identify and localize objects** within images, even in cluttered scenes.  
  Example: Detecting elephants (*tuskers*) in images with varying poses and backgrounds.

- **Clustering Results:**  
  Used unsupervised techniques like **k-means** to find patterns within categories:
  - **Tuskers:** Grouped images into side views, front views, etc.
  - **Aircraft:** Clusters based on whether planes were flying or grounded.

## 5. Future Directions (as of 2009)

### Expanding Scale:
- Targeting **50 million images** covering **all 80,000 WordNet synsets**.

### New Annotations:
- Adding:
  - **Object segmentation** (outlining objects within images).
  - **Localization** (bounding boxes).
  - **Cross-synset references** (linking related images across categories).

### Broad Impact:
- Envisioned to support not just computer vision but also **cognitive science** (e.g., studying how humans categorize objects).

---

This README provides a detailed breakdown of ImageNet’s significance, methodology, and impact. Let me know if you’d like any modifications or additions!
