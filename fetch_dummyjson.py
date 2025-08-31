import requests, json

# Fetch 60 products from DummyJSON
resp = requests.get("https://dummyjson.com/products?limit=60")
data = resp.json()["products"]

# Normalize for our matcher (id, name, category, image_url, price)
products = []
for p in data:
    products.append({
        "id": p["id"],
        "name": p.get("title", "Unknown"),
        "category": p.get("category", "Unknown"),
        "image_url": p.get("thumbnail", ""),
        "price": p.get("price", None)
    })

# Save with the same filename app.py expects
with open("product_db_real.json", "w") as f:
    json.dump(products, f, indent=2)

print(f"Saved {len(products)} products to product_db_real.json")
