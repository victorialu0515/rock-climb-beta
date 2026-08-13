import cv2, json, numpy as np

def draw_detection(image_path, rocks_json, out_path, highlight_track=None, max_width=700):
    img = cv2.imread(image_path)
    rocks = json.load(open(rocks_json))["rocks"]
    for r in rocks:
        x,y,w,h = r["bbox"]
        b,g,rr = [int(c) for c in r["colour"][::-1]]  # colour stored as RGB, convert to BGR for cv2
        thickness = 4 if (highlight_track is not None and r["track"]==highlight_track) else 2
        alpha_color = (b,g,rr)
        cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), alpha_color, thickness)
    h0, w0 = img.shape[:2]
    scale = max_width / w0
    img = cv2.resize(img, (max_width, int(h0*scale)))
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

def draw_pose(image_path, pose, out_path, max_width=700):
    img = cv2.imread(image_path)
    pose = {int(k): v for k, v in pose.items()}
    skel = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),(7,9),(8,10)]
    colors = [(255,0,127),(254,37,103),(251,77,77),(248,115,51),(242,149,25),(235,180,0),
              (227,205,24),(217,226,50),(206,242,76),(193,251,102),(179,254,128),(165,251,152)]
    for i,(a,b) in enumerate(skel):
        pa, pb = pose.get(a,[0,0]), pose.get(b,[0,0])
        if pa[0]>0 and pb[0]>0:
            cv2.line(img, tuple(map(int,pa)), tuple(map(int,pb)), colors[i], 8, cv2.LINE_AA)
    for j in [5,6,7,8,9,10,11,12,13,14,15,16]:
        p = pose.get(j,[0,0])
        if p[0]>0:
            cv2.circle(img, tuple(map(int,p)), 12, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(map(int,p)), 12, (40,40,40), 2, cv2.LINE_AA)
    h0,w0 = img.shape[:2]
    scale = max_width/w0
    img = cv2.resize(img, (max_width, int(h0*scale)))
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

def draw_nextmove(image_path, pose_in, rocks_json, result_json, out_path, max_width=700):
    PAD_T, PAD_B, PAD_L, PAD_R = 70, 30, 30, 320
    img = cv2.imread(image_path)
    img = cv2.copyMakeBorder(img, PAD_T, PAD_B, PAD_L, PAD_R, cv2.BORDER_CONSTANT, value=(255,255,255))
    pose = {int(k): [v[0]+PAD_L, v[1]+PAD_T] for k, v in pose_in.items()}
    def shift_bbox(bbox):
        return [bbox[0]+PAD_L, bbox[1]+PAD_T, bbox[2], bbox[3]]
    rocks = {r["id"]: {**r, "bbox": shift_bbox(r["bbox"])} for r in json.load(open(rocks_json))["rocks"]}
    result = json.load(open(result_json))
    result["current"] = {k: {**v, "bbox": shift_bbox(v["bbox"])} for k,v in result["current"].items()}

    # draw full skeleton (dim, with dark outline so it reads against any background)
    skel = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(6,8),(7,9),(8,10)]
    for (a,b) in skel:
        pa, pb = pose.get(a,[0,0]), pose.get(b,[0,0])
        if pa[0]>0 and pb[0]>0:
            cv2.line(img, tuple(map(int,pa)), tuple(map(int,pb)), (60,60,60), 9, cv2.LINE_AA)
            cv2.line(img, tuple(map(int,pa)), tuple(map(int,pb)), (255,255,255), 4, cv2.LINE_AA)

    # highlight current rocks (cyan)
    for limb, info in result["current"].items():
        x,y,w,h = info["bbox"]
        cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (255,255,0), 4)

    # rank all candidates, dedupe by (limb, rock) keeping best score
    best = {}
    for c in result["candidates"]:
        key = (c["moving_limb"], c["new_rock_id"])
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c
    ranked = sorted(best.values(), key=lambda c: -c["score"])

    # dim runner-up candidates (thin, no label) so the #1 pick stands out
    for c in ranked[1:4]:
        rock = rocks[c["new_rock_id"]]
        x,y,w,h = rock["bbox"]
        cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (0,165,255), 2)

    # highlight the single top-ranked move
    top = ranked[0]
    rock = rocks[top["new_rock_id"]]
    x,y,w,h = rock["bbox"]
    color = (0,0,220)
    cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), color, 6)
    cur = result["current"][top["moving_limb"]]["bbox"]
    cv2.arrowedLine(img, (int(cur[0]),int(cur[1])), (int(x),int(y)), color, 4, tipLength=0.06)
    label = f"BEST MOVE: {top['moving_limb']}  (score {top['score']:.0f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    label_y = max(int(y-h/2-16), 30)
    label_x = max(min(int(x-w/2)-20, img.shape[1]-tw-20), 10)
    cv2.rectangle(img, (label_x-6, label_y-th-8), (label_x+tw+6, label_y+6), (255,255,255), -1)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    h0,w0 = img.shape[:2]
    scale = max_width/w0
    img = cv2.resize(img, (max_width, int(h0*scale)))
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


# Diamond wall scene
draw_detection("/mnt/user-data/uploads/IMG_2271_time_9.jpg", "demo_assets/IMG_2271_time_9_rocks.json", "demo_assets/diamond_detection.jpg", highlight_track=0)
draw_pose("/mnt/user-data/uploads/IMG_2271_time_9.jpg", json.load(open("demo_assets/pose1.json")), "demo_assets/diamond_pose.jpg")
draw_nextmove("/mnt/user-data/uploads/IMG_2271_time_9.jpg", json.load(open("demo_assets/pose1.json")), "demo_assets/IMG_2271_time_9_rocks.json", "demo_assets/diamond_result.json", "demo_assets/diamond_nextmove.jpg")

# Willtopia wall scene (clean empty wall for detection, person photo for pose/move)
draw_detection("/mnt/user-data/uploads/IMG_2264_time_41.jpg", "demo_assets/IMG_2264_time_41_rocks.json", "demo_assets/willtopia_detection.jpg", highlight_track=0)
draw_pose("/mnt/user-data/uploads/IMG_2264_time_19.jpg", json.load(open("demo_assets/pose2.json")), "demo_assets/willtopia_pose.jpg")
draw_nextmove("/mnt/user-data/uploads/IMG_2264_time_19.jpg", json.load(open("demo_assets/pose2.json")), "demo_assets/IMG_2264_time_41_rocks.json", "demo_assets/willtopia_result.json", "demo_assets/willtopia_nextmove.jpg")

print("done")
