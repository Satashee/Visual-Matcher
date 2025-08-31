import cv2
import numpy as np
import urllib.request

# ===================== Tunables =====================
MAX_SIDE = 512              # resize longest side
ORB_FEATURES = 1400
AKAZE_THRESH = 1e-4         # AKAZE default
RATIO = 0.80                # Lowe ratio
RANSAC_REPROJ = 5.0
TIMEOUT = 3                 # seconds per image fetch
H_BINS, S_BINS = 50, 60
LBP_BINS = 256
CANDIDATE_TOPK = 25         # prefilter size before ORB+RANSAC

# Blend weights (sum near 1.0)
W_GEOM = 0.55               # ORB/AKAZE + RANSAC inliers (structure)
W_COLOR = 0.30              # HSV hist on foreground mask
W_TEXTURE = 0.10            # LBP texture on foreground
W_PHASH = 0.05              # perceptual hash

# ===================== I/O helpers =====================
def fetch_image(url: str, timeout: int = TIMEOUT):
    if not url:
        return None
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def preprocess(img_bgr):
    """Resize + color constancy (gray-world) + mild denoise."""
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Gray-world white balance
    avg = img_bgr.reshape(-1, 3).mean(0)
    gain = avg.mean() / (avg + 1e-6)
    img_bgr = np.clip(img_bgr * gain, 0, 255).astype(np.uint8)

    # Slight denoise
    img_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=50, sigmaSpace=50)
    return img_bgr

# ===================== Foreground mask =====================
def quick_grabcut_mask(img_bgr):
    """Fast GrabCut init by full-image rectangle; returns mask {0,1}."""
    h, w = img_bgr.shape[:2]
    # shrink rectangle slightly to avoid including borders/background
    pad = int(0.06 * max(h, w))
    rect = (pad, pad, max(1, w - 2*pad), max(1, h - 2*pad))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        return fg
    except Exception:
        return np.ones((h, w), np.uint8)  # fall back to all-foreground

def apply_mask(img_bgr, mask01):
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask01)

# ===================== Feature extractors =====================
def clahe_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def extract_orb(img_bgr):
    if img_bgr is None:
        return [], None
    gray = clahe_gray(img_bgr)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=7)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 200:
        # Fallback: AKAZE if ORB is weak
        akaze = cv2.AKAZE_create(threshold=AKAZE_THRESH)
        kp2, des2 = akaze.detectAndCompute(gray, None)
        if des2 is not None and len(kp2) > len(kp or []):
            return kp2, des2
    return kp or [], des

def extract_hs_hist(img_bgr, mask01=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mask01 is not None:
        mask = (mask01 * 255).astype(np.uint8)
    else:
        mask = None
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def extract_lbp_hist(img_bgr, mask01=None):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    radius = 1
    if h <= 2*radius or w <= 2*radius:
        return None
    center = gray[radius:-radius, radius:-radius].astype(np.int16)
    lbp = np.zeros_like(center, dtype=np.uint8)
    for idx, (dy, dx) in enumerate([(-1,0),(0,1),(1,0),(0,-1),(-1,-1),(1,1),(-1,1),(1,-1)]):
        neigh = gray[radius+dy:h-radius+dy, radius+dx:w-radius+dx].astype(np.int16)
        lbp |= ((neigh >= center) << idx).astype(np.uint8)
    if mask01 is not None:
        m = mask01[radius:-radius, radius:-radius]
        vals = lbp[m.astype(bool)]
    else:
        vals = lbp.ravel()
    hist, _ = np.histogram(vals, bins=LBP_BINS, range=(0,256), density=True)
    return hist.astype(np.float32)

def phash(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))
    dct_low = dct[:8, :8]
    med = np.median(dct_low)
    bits = (dct_low > med).astype(np.uint8).flatten()
    val = np.uint64(0)
    for b in bits:
        val = (val << np.uint64(1)) | np.uint64(b)
    return val

# ===================== Similarities =====================
def orb_geom_similarity(img_q, img_d):
    kp_q, des_q = extract_orb(img_q)
    kp_d, des_d = extract_orb(img_d)
    if des_q is None or des_d is None or not kp_q:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_q, des_d, k=2)
    good = [m for m, n in matches if m.distance < RATIO * n.distance]
    if len(good) >= 4:
        src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp_d[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
        inliers = int(mask.sum()) if mask is not None else 0
    else:
        inliers = len(good)
    return min(inliers / max(1, len(kp_q)), 1.0)

def bhatta_sim(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)  # 0 best
    return max(0.0, 1.0 - min(1.0, float(d)))

def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    num = float((v1 * v2).sum())
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return max(0.0, min(num / den, 1.0))

def phash_sim(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    x = int(h1 ^ h2)
    dist = 0
    while x:
        x &= x - 1
        dist += 1
    return max(0.0, 1.0 - dist / 64.0)

def combined_score(s_geom, s_col, s_tex, s_ph):
    return W_GEOM*s_geom + W_COLOR*s_col + W_TEXTURE*s_tex + W_PHASH*s_ph

# ===================== Product index =====================
class ProductEntry:
    __slots__ = ("p", "img", "mask", "hist_hs", "lbp", "ph", "ok")
    def __init__(self, p, img, mask, hist_hs, lbp, ph, ok):
        self.p = p
        self.img = img
        self.mask = mask
        self.hist_hs = hist_hs
        self.lbp = lbp
        self.ph = ph
        self.ok = ok

class ProductIndex:
    def __init__(self, entries):
        self.entries = entries

def build_product_index(product_db, verbose=True):
    entries, ok = [], 0
    for p in product_db:
        url = p.get("image_url") or p.get("thumbnail") or ""
        raw = fetch_image(url)
        if raw is None:
            entries.append(ProductEntry(p, None, None, None, None, None, False))
            continue
        img = preprocess(raw)
        mask = quick_grabcut_mask(img)
        fgi = apply_mask(img, mask)

        hist = extract_hs_hist(fgi, mask)
        lbp = extract_lbp_hist(fgi, mask)
        ph = phash(fgi)

        entries.append(ProductEntry(p, img, mask, hist, lbp, ph, True))
        ok += 1

    if verbose:
        print(f"[matcher] Indexed {ok}/{len(product_db)} images with features.")
    return ProductIndex(entries)

# ===================== Query =====================
def match_query(query_img_bgr, index: ProductIndex, top_k=30):
    if query_img_bgr is None or not index.entries:
        return []

    q_img = preprocess(query_img_bgr)
    q_mask = quick_grabcut_mask(q_img)
    q_fgi = apply_mask(q_img, q_mask)

    q_hist = extract_hs_hist(q_fgi, q_mask)
    q_lbp = extract_lbp_hist(q_fgi, q_mask)
    q_ph = phash(q_fgi)

    # ---------- Stage 1: fast prefilter by color ----------
    coarse = []
    for e in index.entries:
        if not e.ok:
            continue
        s_col = bhatta_sim(q_hist, e.hist_hs)
        # quick combo with texture + phash (cheap)
        s_tex = cosine_sim(q_lbp, e.lbp)
        s_ph = phash_sim(q_ph, e.ph)
        s_coarse = 0.6*s_col + 0.25*s_tex + 0.15*s_ph
        coarse.append((s_coarse, e))
    coarse.sort(key=lambda t: t[0], reverse=True)
    candidates = [e for _, e in coarse[:CANDIDATE_TOPK]]

    # ---------- Stage 2: precise geometry on candidates ----------
    results = []
    for e in candidates:
        s_col = bhatta_sim(q_hist, e.hist_hs)
        s_tex = cosine_sim(q_lbp, e.lbp)
        s_ph = phash_sim(q_ph, e.ph)
        s_geom = orb_geom_similarity(q_fgi, apply_mask(e.img, e.mask))
        score = combined_score(s_geom, s_col, s_tex, s_ph)
        p = e.p
        results.append({
            "id": p.get("id"),
            "name": p.get("name", "Unknown"),
            "category": p.get("category", "Unknown"),
            "image_url": p.get("image_url") or p.get("thumbnail") or "",
            "price": p.get("price"),
            "score": round(score * 100.0, 2)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
