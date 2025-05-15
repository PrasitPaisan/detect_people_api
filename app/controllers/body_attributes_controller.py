from app.models.request_model import BodyAttributes
from fastapi import APIRouter

router = APIRouter()

@router.post("/get/tk_API/body_attributes")
def read_item(request: BodyAttributes):
    body_attributes = {
        "results": [
            {
                "code": 0,
                "error": "",
                "status": "OK"
            }
        ],
        "responses": [
            {
                "objects": [
                    {
                        "object_info": {
                            "type": "OBJECT_PEDESTRIAN",
                            "face": None,
                            "pedestrian": {
                                "quality": 0,
                                "rectangle": {
                                    "vertices": [
                                        {
                                            "x": 0,
                                            "y": 4
                                        },
                                        {
                                            "x": 388,
                                            "y": 776
                                        }
                                    ]
                                },
                                "track_id": "0",
                                "attributes_with_score": {
                                    "age_lower_limit": {
                                        "type": "CLASSIFICATION",
                                        "category": "AGE_LOWER_LIMIT",
                                        "value": 18,
                                        "roi": None
                                    },
                                    "age_up_limit": {
                                        "type": "CLASSIFICATION",
                                        "category": "AGE_UP_LIMIT",
                                        "value": 59,
                                        "roi": None
                                    },
                                    "cap_style": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_CRASH_HELMET",
                                        "value": 0.9875618,
                                        "roi": None
                                    },
                                    "coat_color": {
                                        "type": "CLASSIFICATION",
                                        "category": "BLACK",
                                        "value": 0.9922662,
                                        "roi": None
                                    },
                                    "coat_length": {
                                        "type": "CLASSIFICATION",
                                        "category": "SHORT_SLEEVE",
                                        "value": 0.9999188,
                                        "roi": None
                                    },
                                    "gender_code": {
                                        "type": "CLASSIFICATION",
                                        "category": "MALE",
                                        "value": 0.8538243,
                                        "roi": None
                                    },
                                    "glass_style": {
                                        "type": "CLASSIFICATION",
                                        "category": "GLASSES_STYLE_TYPE_WITHOUT",
                                        "value": 0.94858325,
                                        "roi": None
                                    },
                                    "hair_style": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_SHORT",
                                        "value": 0.93048984,
                                        "roi": None
                                    },
                                    "st_age": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_ADULT",
                                        "value": 0.90788525,
                                        "roi": None
                                    },
                                    "st_coat_pattern": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_PURE",
                                        "value": 0.9620502,
                                        "roi": None
                                    },
                                    "st_fishing": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_FISHING",
                                        "value": 0.99999976,
                                        "roi": None
                                    },
                                    "st_glove": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_GLOVE_WITHOUT",
                                        "value": 0.99916893,
                                        "roi": None
                                    },
                                    "st_glove_v2": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_GLOVE_WITHOUT",
                                        "value": 0.99916893,
                                        "roi": None
                                    },
                                    "st_hold_object_in_front": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_HOLD_OBJECT_IN_FRONT_WITHOUT",
                                        "value": 0.9892161,
                                        "roi": None
                                    },
                                    "st_hold_object_in_front_v2": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_HOLD_OBJECT_IN_FRONT_WITHOUT",
                                        "value": 0.9892161,
                                        "roi": None
                                    },
                                    "st_oxygen_bottle": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_OXYGEN_BOTTLE_AGNOSTIC",
                                        "value": 0.9305176,
                                        "roi": None
                                    },
                                    "st_pedestrian_angle": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_FRONT",
                                        "value": 0.98465145,
                                        "roi": None
                                    },
                                    "st_phone_status": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_NORMAL",
                                        "value": 0.9756639,
                                        "roi": None
                                    },
                                    "st_pose": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_SIT_ON_THE_VEHICLE",
                                        "value": 0.98935705,
                                        "roi": None
                                    },
                                    "st_process_internal_detect_confidence": {
                                        "type": "CLASSIFICATION",
                                        "category": "INTERNAL_DETECT_CONFIDENCE",
                                        "value": 0.61850697,
                                        "roi": None
                                    },
                                    "st_process_internal_ped_quality": {
                                        "type": "CLASSIFICATION",
                                        "category": "BLUR",
                                        "value": 0.9999941,
                                        "roi": None
                                    },
                                    "st_reflective_clothes": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_REFLECTIVE_CLOTHES_WITHOUT",
                                        "value": 0.99968904,
                                        "roi": None
                                    },
                                    "st_smoking": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_SMOKING_WITHOUT",
                                        "value": 0.93009865,
                                        "roi": None
                                    },
                                    "st_trousers_pattern": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_PURE",
                                        "value": 0.9805157,
                                        "roi": None
                                    },
                                    "st_umbrella": {
                                        "type": "CLASSIFICATION",
                                        "category": "ST_UMBRELLA_WITHOUT",
                                        "value": 0.9999932,
                                        "roi": None
                                    }
                                },
                                "pedestrian_score": 0.9694756
                            },
                            "automobile": None,
                            "human_powered_vehicle": None,
                            "cyclist": None,
                            "crowd": None,
                            "event": None,
                            "portrait_image_location": None,
                            "object_id": "0",
                            "associations": [],
                            "algo": None,
                            "diagnosis": None,
                            "carplate": None,
                            "watercraft": None,
                            "trajectory": None
                        },
                        "feature": {
                            "type": "pedestrian",
                            "version": 10500,
                            "blob": {request.image_path}
                        }
                    }
                ]
            }
        ]
    }
    return body_attributes
