from flask import Flask, render_template, request, flash
import os, uuid, json, cv2, urllib.request, numpy as np
from matcher import build_product_index, match_query   # 👈 NEW matcher functions

# ---- CONFIG ----
DATA_PATH = "product_db_real.json"   # JSON in project root
UPLOAD_FOLDER = "static/uploads"

app = Flask(__name__)
app.secret_key = "dev"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Normalize dataset ----
REQUIRED_FIELDS = ["id", "name", "category", "image_url"]  # price optional

def load_and_normalize_products(path):
    with open(path, "r") as f:
        raw = json.load(f)

    products = []
    dropped = 0
    for i, p in enumerate(raw, 1):
        if not any(k in p and str(p[k]).strip() for k in ["image_url","thumbnail"]):
            dropped += 1
            continue

        price_val = p.get("price", None)
        try:
            price = float(price_val) if price_val not in ("", None) else None
        except Exception:
            price = None

        products.append({
            "id": p.get("id", i),
            "name": str(p.get("name") or p.get("title") or "Unknown"),
            "category": str(p.get("category", "Unknown")),
            "image_url": str(p.get("image_url") or p.get("thumbnail") or ""),
            "price": price
        })

    print(f"Loaded {len(products)} valid products from {path} (dropped {dropped})")
    return products

PRODUCTS = load_and_normalize_products(DATA_PATH)

# ---- Build index (precompute ORB + histogram for each product) ----
INDEX = build_product_index(PRODUCTS)

# ---- Helpers ----
def url_to_image(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        app.logger.error(f"URL fetch error: {e}")
        return None

def decode_file_to_cv2(file_storage):
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    file_storage.stream.seek(0)
    return img

# ---- Routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            source = request.form.get("source", "upload")
            uploaded_path = None
            img = None

            if source == "url":
                url = (request.form.get("image_url") or "").strip()
                if not url:
                    flash("Please paste an image URL.", "error")
                    return render_template("index.html")
                img = url_to_image(url)
                if img is None:
                    flash("Could not fetch/decode image from URL.", "error")
                    return render_template("index.html")

            else:  # file upload
                file = request.files.get("image")
                if not file or file.filename == "":
                    flash("Please choose an image file to upload.", "error")
                    return render_template("index.html")
                img = decode_file_to_cv2(file)
                if img is None:
                    flash("Unsupported image format. Use JPG/PNG/WEBP.", "error")
                    return render_template("index.html")

            # Save preview image for UI
            temp_name = f"{uuid.uuid4().hex}.jpg"
            uploaded_path = os.path.join(UPLOAD_FOLDER, temp_name)
            cv2.imwrite(uploaded_path, img)

            # ---- Real matcher ----
            if not INDEX.entries:
                flash("No products indexed. Check dataset image URLs.", "error")
                return render_template("index.html")

            results = match_query(img, INDEX, top_k=30)

            if not results:
                flash("No results (feature extraction failed?). Try another image.", "error")
                return render_template("index.html")

            return render_template("results.html", results=results, uploaded_image=uploaded_path)

        except Exception as e:
            import traceback
            app.logger.error("ERROR during POST:\n" + traceback.format_exc())
            flash(f"Something went wrong: {e}", "error")
            return render_template("index.html")

    return render_template("index.html")

if __name__ == "__main__":
    print(f"Indexed {len(INDEX.entries)} product images for matching.")
    app.run(debug=True)
