import uuid
from datetime import datetime

def generate_data(snap_path, ovew_path, ovem_path_roi, timestamp):

    data = {
        "body_det_id": str(uuid.uuid4()),
        "snapshot_image": snap_path,
        "overview_image": ovew_path,
        "overview_image_roi": ovem_path_roi,
        "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return data
