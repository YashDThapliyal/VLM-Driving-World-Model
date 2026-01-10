from dataclasses import dataclass
from typing import Literal


GoStraight = Literal["safe", "unsafe", "unknown"]
Brake = Literal["required", "not_required", "uncertain"]
TrafficLightState = Literal["red", "yellow", "green", "none"]


@dataclass
class AffordanceLabel:
    """
    Human label for a single frame's affordances.
    frame_id is relative to frames_root, e.g.: "my_drive_01/frame_0005.jpg"
    """
    frame_id: str
    go_straight: GoStraight
    brake: Brake
    pedestrian_present: bool
    lead_vehicle_present: bool
    traffic_light_state: TrafficLightState


def affordance_schema_description() -> str:
    """
    Returns a description of the affordance schema for prompting a VLM.
    This is written as if you're explaining the task to another AI model.
    """
    return """
You are an autonomous driving affordance estimator for a forward-facing camera on an ego vehicle.

Given a single road scene image, you must output STRICT JSON with the following fields:

- "go_straight": one of "safe", "unsafe", or "unknown".
   * "safe" means the ego vehicle can continue straight without immediate collision or clear rule violation.
   * "unsafe" means continuing straight would likely cause a collision, run a red light, hit a pedestrian, or otherwise be clearly unsafe.
   * "unknown" means the image is too ambiguous to decide.

- "brake": one of "required", "not_required", or "uncertain".
   * "required" means the ego vehicle should start braking now or already be braking to remain safe and compliant.
   * "not_required" means braking is not immediately necessary from a safety perspective.
   * "uncertain" means you cannot confidently decide.

- "pedestrian_present": true or false.
   * true if there is at least one pedestrian or person whose motion could interact with the ego vehicle's path.

- "lead_vehicle_present": true or false.
   * true if there is a vehicle directly ahead in the same lane or obviously acting as the lead vehicle.

- "traffic_light_state": one of "red", "yellow", "green", or "none".
   * Use "none" if there is no relevant traffic light visible or it is not clearly interpretable.

You MUST reason from the perspective of the ego vehicle in the scene.
When in doubt about safety, be conservative: prefer "unsafe" for go_straight and "required" for brake.
Return ONLY JSON, no extra text.
""".strip()
