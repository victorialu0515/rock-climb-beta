import json, math, numpy as np, joblib, pandas as pd, copy

def closest_rock(point, rocks):
    best, bestd = None, float("inf")
    for r in rocks:
        x,y,w,h = r["bbox"]
        d = math.hypot(point[0]-x, point[1]-y)
        if d < bestd:
            bestd = d; best = r
    return best, bestd

def feature_engineering(df):
    df = df.copy()
    df["left_foot_delta_x"] = (df["leftFoot_x"] - df["leftHand_x"])/df["person_size"]
    df["left_foot_delta_y"] = (df["leftFoot_y"] - df["leftHand_y"])/df["person_size"]
    df["left_hand_delta_x"] = (df["leftHand_x"] - df["rightHand_x"])/df["person_size"]
    df["left_hand_delta_y"] = (df["leftHand_y"] - df["rightHand_y"])/df["person_size"]
    df["right_foot_delta_x"] = (df["rightFoot_x"] - df["rightHand_x"])/df["person_size"]
    df["right_foot_delta_y"] = (df["rightFoot_y"] - df["rightHand_y"])/df["person_size"]
    df["right_hand_delta_x"] = (df["rightHand_x"] - df["nextRightHand_x"])/df["person_size"]
    df["right_hand_delta_y"] = (df["rightHand_y"] - df["nextRightHand_y"])/df["person_size"]
    snake = {"leftHand":"left_hand","rightHand":"right_hand","leftFoot":"left_foot","rightFoot":"right_foot"}
    for part in ["leftHand","rightHand","leftFoot","rightFoot"]:
        df[f"{snake[part]}_relative_x"] = (df[f"{part}_x"] - df["person_center_x"])/df["person_size"]
        df[f"{snake[part]}_relative_y"] = (df[f"{part}_y"] - df["person_center_y"])/df["person_size"]
    return df

def run(pose_file, rocks_file, out_name):
    pose = {int(k): v for k, v in json.load(open(pose_file)).items()}
    result_dict = json.load(open(rocks_file))
    rocks = result_dict["rocks"]

    landmarks = np.array([pose.get(i, [0,0]) for i in range(17)], dtype=float)
    xarr, yarr = landmarks[:,0], landmarks[:,1]
    landmarks[landmarks[:,0]<=0,0] = xarr[xarr>0].mean()
    landmarks[landmarks[:,1]<=0,1] = yarr[yarr>0].mean()
    person_size = max(landmarks[:,0].max()-landmarks[:,0].min(), landmarks[:,1].max()-landmarks[:,1].min())

    limb_pts = {"leftHand": tuple(pose[9]), "rightHand": tuple(pose[10]),
                "leftFoot": tuple(pose[15]), "rightFoot": tuple(pose[16])}
    current = {}
    for name, pt in limb_pts.items():
        rock, dist = closest_rock(pt, rocks)
        current[name] = rock
        print(f"  {name}: pose_pt={pt} -> rock id={rock['id']} track={rock['track']} dist={dist:.0f}px")

    track_votes = {}
    for r in current.values():
        track_votes[r["track"]] = track_votes.get(r["track"], 0) + 1
    track_chosen = max(track_votes, key=track_votes.get)
    print("  track_chosen (majority vote):", track_chosen, track_votes)

    centerx = (min(pose[5][0], pose[6][0]) + max(pose[11][0], pose[12][0])) / 2
    centery = (min(pose[5][1], pose[6][1]) + max(pose[11][1], pose[12][1])) / 2

    rocks_track = [r for r in rocks if r["track"] == track_chosen]
    current_ids = {v["id"] for v in current.values()}
    limb_order = ["leftHand","rightHand","leftFoot","rightFoot"]
    rows = []
    for limb in limb_order:
        is_hand = "Hand" in limb
        cx0, cy0, cw0, ch0 = current[limb]["bbox"]
        for j in rocks_track:
            x,y,w,h = j["bbox"]
            if j["id"] == current[limb]["id"]:
                continue
            distance = math.hypot(centerx-x, centery-y)
            if is_hand:
                ok = distance < person_size*0.8 and j["id"] not in current_ids and cy0 - y > person_size*-0.05
            else:
                ok = distance < person_size*0.7 and j["id"] not in current_ids and y - centery > person_size*-0.04
            if ok:
                row = {"moving_limb": limb, "new_rock_id": j["id"], "new_rock_xy": (x,y),
                       "person_center_x": centerx, "person_center_y": centery, "person_size": person_size}
                for l2 in limb_order:
                    bx,by = current[l2]["bbox"][0], current[l2]["bbox"][1]
                    row[f"{l2}_x"], row[f"{l2}_y"] = bx, by
                    row[f"next{l2[0].upper()+l2[1:]}_x"], row[f"next{l2[0].upper()+l2[1:]}_y"] = bx, by
                row[f"next{limb[0].upper()+limb[1:]}_x"] = x
                row[f"next{limb[0].upper()+limb[1:]}_y"] = y
                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  candidate moves found: {len(df)}")
    if len(df) == 0:
        return
    df = feature_engineering(df)
    model = joblib.load("/mnt/user-data/uploads/linear_model.joblib")
    predictors = sorted([c for c in df.columns if "_relative_" in c or "_delta_" in c])
    X = df[predictors]
    df["score"] = model.predict(X)
    df_sorted = df.sort_values("score", ascending=False)
    print(df_sorted[["moving_limb","new_rock_id","score"]].head(8).to_string(index=False))

    out = {
        "track_chosen": track_chosen,
        "current": {k: {"id": v["id"], "bbox": v["bbox"]} for k,v in current.items()},
        "candidates": df_sorted[["moving_limb","new_rock_id","score"]].to_dict("records"),
        "person_center": [centerx, centery], "person_size": person_size,
    }
    json.dump(out, open(f"demo_assets/{out_name}", "w"), indent=2, default=str)

print("=== diamond_wall ===")
run("demo_assets/pose1.json", "demo_assets/IMG_2271_time_9_rocks.json", "diamond_result.json")
print("\n=== willtopia_wall ===")
run("demo_assets/pose2.json", "demo_assets/IMG_2264_time_41_rocks.json", "willtopia_result.json")
