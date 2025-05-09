import cv2
import math

def extract_detected_objects(results, roi_intrusion):
    persons = []
    vehicles = []

    if results.boxes is not None:
        for i in range(len(results.boxes)):
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[i].cpu().numpy())
            cls_id = int(results.boxes.cls[i].cpu().item())
            track_id = int(results.boxes.id[i].cpu().item(
            )) if results.boxes.id is not None else None
            conf = float(results.boxes.conf[i].cpu().item())

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            inside = cv2.pointPolygonTest(roi_intrusion, center, False) >= 0

            if not inside:
                continue

            if cls_id == 0:
                persons.append(
                    {"box": (x1, y1, x2, y2), "id": track_id, "conf": conf})
            if cls_id in {1, 3}:  # bicycle or motorcycle
                vehicles.append(
                    {"box": (x1, y1, x2, y2), "id": track_id, "conf": conf})

    return persons, vehicles

def find_closest_vehicle(person, vehicles, used_vehicle_ids, max_distance=200):
    px1, py1, px2, py2 = person["box"]
    pid = person["id"]
    p_center = (int((px1+px2)//2), int((py1+py2)//2))

    closest_vehicle = None
    min_distance = float("inf")

    for vehicle in vehicles:
        vid = vehicle["id"]
        if vid in used_vehicle_ids:
            continue

        vx1, vy1, vx2, vy2 = vehicle["box"]
        v_center = (int((vx1+vx2)//2), int((vy1+vy2)//2))

        dist = math.hypot(p_center[0] - v_center[0], p_center[1] - v_center[1])

        if dist < min_distance and dist < max_distance:
            min_distance = dist
            closest_vehicle = vehicle

    return closest_vehicle, pid, dist

def draw_detection(frame, showframe, roi, person_box, vehicle_box):
    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = vehicle_box
    x1 = min(px1, vx1)
    y1 = min(py1, vy1)
    x2 = max(px2, vx2)
    y2 = max(py2, vy2)

    cv2.polylines(showframe, [roi],
                  isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.rectangle(showframe, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(showframe, f"Rider", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    snapshot = frame[y1:y2, x1:x2]

    return snapshot
