# Visual Product Matcher

## Overview
The **Visual Product Matcher** is a Flask-based web application that allows users to upload an image or paste an image URL to find visually similar products from a product catalog. It combines several classical computer vision techniques to compute similarity scores without relying on heavy machine learning frameworks like TensorFlow or PyTorch.  

This project was built as part of a technical assessment but can serve as a solid reference or starting point for image search engines, e-commerce recommendation systems, or educational demonstrations of computer vision.

---

## Key Features
- **Image Upload & URL Support**: Users can either upload a local image file or paste an external image link.  
- **Product Catalog**: Dataset of 60+ products (from DummyJSON or Unsplash-style sources) with metadata: `id`, `name`, `category`, `price`, and `image_url`.  
- **Feature Extraction**:
  - ORB + AKAZE keypoints for geometric/structural similarity.
  - HSV color histograms for color similarity.
  - LBP (Local Binary Pattern) histograms for texture.
  - Perceptual hash (pHash) for global appearance similarity.
- **Foreground Segmentation**: Uses GrabCut to reduce background noise and focus on the product.  
- **Similarity Scoring**: Weighted blend of structural, color, texture, and global appearance metrics.  
- **Interactive UI**: Responsive design with TailwindCSS. Product cards include hover effects, pricing, and a slider filter for adjusting minimum similarity.  
- **Two-Stage Matching**:  
  - Stage 1: Fast prefilter using color/texture.  
  - Stage 2: Detailed geometric (ORB+RANSAC) matching on top candidates.  

---

## Tech Stack
- **Backend**: Flask (Python)  
- **Frontend**: TailwindCSS + Jinja templates  
- **Computer Vision**: OpenCV, NumPy  
- **Dataset**: JSON file with product metadata and image URLs (from DummyJSON)  

---
## Project Structure
 ```
 Unthinkable-Solution/
├── app.py                     # Main Flask app (routes, server logic)
├── matcher.py                 # Similarity logic (ORB + color + texture + pHash)
├── product_db_real.json       # Product dataset (DummyJSON products, id/name/category/image_url/price)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── static/                    # Static files (served directly)
│   └── uploads/               # Uploaded images stored temporarily
│
└── templates/                 # HTML templates (Jinja2 + TailwindCSS)
    ├── index.html             # Homepage: form for upload/URL
    └── results.html           # Results page: product grid with similarity scores
```

## How It Works
1. On startup, the app loads `product_db_real.json` and pre-computes product features (ORB, HSV hist, LBP, pHash).  
2. When a user uploads or links an image, its features are extracted and compared with cached product features.  
3. A blended similarity score (0–100%) is computed for each product.  
4. Results are sorted and displayed in a grid, showing product image, name, category, price, and similarity percentage.  
5. The user can adjust a slider to filter results by minimum similarity.

---

## Installation & Running Locally
1. Clone this repository.  
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Flask app:
  ```
    python app.py
  ```
4. Open the app in your browser at:
    ```
    http://127.0.0.1:500
    ```