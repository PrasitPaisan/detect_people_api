import uuid

def generate_data(snap_path, ovew_path):

    data = {
        "helmet_detection_log_uuid": str(uuid.uuid4()),
        "ai_camera_uuid": str(uuid.uuid4()),
        "snapshot_image": snap_path,
        "overview_image": ovew_path,
        "body_attribute": {}
    }
    return data
