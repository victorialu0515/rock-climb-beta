import numpy as np, cv2, json
from ultralytics import YOLO
from sklearn.cluster import KMeans

def getColour(image_path, result):
    colours = []
    imageArray = cv2.imread(image_path)
    for i in result:
        x, y, w, h = i
        img = imageArray[int(y-h/3):int(y+h/3), int(x-w/3):int(x+w/3)]
        if img.size == 0:
            colours.append(np.array([128.,128.,128.]))
            continue
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        colours.append(dominant[::-1])  # BGR->RGB
    return colours

def clustering(numClusters, colours):
    kmeans = KMeans(n_clusters=numClusters, random_state=42, n_init=10)
    kmeans.fit(colours)
    return list(kmeans.labels_)

def detect(image_path, num_cluster=6, model_path="/mnt/user-data/uploads/best.pt"):
    model = YOLO(model_path)
    results = model(image_path)
    result = [x.tolist() for x in results[0].boxes.xywh]
    colours = getColour(image_path, result)
    tracks = clustering(min(num_cluster, len(result)), colours)
    list_colours = [c.tolist() for c in colours]
    rocks = [
        {"id": i, "bbox": bbox, "colour": colour, "track": int(track)}
        for i, (bbox, colour, track) in enumerate(zip(result, list_colours, tracks))
    ]
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    return {"user_id": "Victoria", "run_id": 0, "track_chosen": 1, "rocks": rocks, "image_width": w, "image_height": h}

for name in ["IMG_2271_time_9", "IMG_2264_time_41", "IMG_2264_time_19"]:
    result = detect(f"/mnt/user-data/uploads/{name}.jpg")
    with open(f"demo_assets/{name}_rocks.json", "w") as f:
        json.dump(result, f, indent=2)
    print(name, "->", len(result["rocks"]), "rocks, image size", result["image_width"], "x", result["image_height"])
