#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdio.h>

// Handle the shared object file with Python --> Will need to run:
// pip install pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

struct BoundingBox {
    float x_min, y_min, x_max, y_max;
    int confidence;
    BoundingBox() : x_min(0), y_min(0), x_max(0), y_max(0), confidence(0) {}

    // Parameterized Constructor (Crucial for pybind11)
    BoundingBox(float x1, float y1, float x2, float y2, int conf) 
        : x_min(x1), y_min(y1), x_max(x2), y_max(y2), confidence(conf) {}
};


// Define a tracker for the Manager to remember boxes over time
struct TrackedBox {
    BoundingBox box;
    int frames_seen; // How long has this specific object been confusing YOLO?
};

int CURRENT_MODEL = 0; // 0 --> YOLO / 1 --> DETR (The big cheese)
const int CONF_THRESHOLD = 50; 
const int FRAME_THRESHOLD = 5; 
const float IOU_THRESHOLD = 0.5; 

// vector of all bad boxes being looked at >:(
vector<TrackedBox> active_weak_targets; 

// IoU Calculator (Remains the same)
float calculateIoU(BoundingBox box1, BoundingBox box2) {
    float x_left = max(box1.x_min, box2.x_min);
    float y_top = max(box1.y_min, box2.y_min);
    float x_right = min(box1.x_max, box2.x_max);
    float y_bottom = min(box1.y_max, box2.y_max);

    if (x_right < x_left || y_bottom < y_top) return 0.0;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float box1_area = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min);
    float box2_area = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min);

    return intersection_area / (box1_area + box2_area - intersection_area);
}

// Pee-Pee
int which_model(const vector<BoundingBox>& current_frame_boxes) {
    
    // If DETR is already engaged, wait for it to finish and reset.
    if (CURRENT_MODEL == 1) {
        return CURRENT_MODEL; 
    }

    // We create a temporary list to hold the targets we are carrying over to the next frame
    vector<TrackedBox> next_generation_targets;
    bool trigger_detr = false;

    // Loop through every object YOLO found in this specific frame
    for (const auto& yolo_box : current_frame_boxes) {
        
        // Only tracking confused boxes:
        if (yolo_box.confidence <= CONF_THRESHOLD && yolo_box.confidence != -1) {
            
            bool matched_existing_target = false;

            for (auto& tracked : active_weak_targets) {
                float iou = calculateIoU(yolo_box, tracked.box);
                if (iou >= IOU_THRESHOLD) {
                    TrackedBox updated_target;
                    updated_target.box = yolo_box;
                    updated_target.frames_seen = tracked.frames_seen + 1;
                    
                    next_generation_targets.push_back(updated_target);
                    matched_existing_target = true;

                    // FOR NOW (to be tested later): CHECK FOR 5 WEAK CONFIDENCES, IF SO, SWITCH TO DETR FOR A BIT
                    if (updated_target.frames_seen >= FRAME_THRESHOLD) {
                        trigger_detr = true;
                    }
                    break; // Stop looking for matches for this specific box
                }
            }

            if (!matched_existing_target) {
                TrackedBox new_target = {yolo_box, 1};
                next_generation_targets.push_back(new_target);
            }
        }
    }

    // Notice: Any old targets that weren't matched this frame are naturally forgotten!
    active_weak_targets = next_generation_targets;

    if (trigger_detr) {
        CURRENT_MODEL = 1;
        cout << "[MANAGER] Unstable target locked for " << FRAME_THRESHOLD << " frames. Engaging DETR." << endl;
        // Get rid of all bounding boxes once DETR appears (The big cheese)
        active_weak_targets.clear(); 
    }

    return CURRENT_MODEL;
}

void reset_model() {
    CURRENT_MODEL = 0;
    active_weak_targets.clear();
}

// No need for main file when doing Shared Object: :P
// int main(int argc, char* argv[]) {
//     // Simulated Example: A frame with multiple objects
//     for (int frame = 1; frame <= 6; frame++) {
//         cout << "--- Frame " << frame << " ---" << endl;
        
//         vector<BoundingBox> frame_data;
        

//         int next_model = which_model(frame_data);
//         cout << "Action: " << (next_model == 0 ? "Continue YOLO" : "Switch to DETR") << "\n\n";
//     }

//     return 0;
// }

namespace py = pybind11;

PYBIND11_MODULE(manager, m){
    py::class_<BoundingBox>(m, "BoundingBox")
    .def(py::init<float, float, float, float, int>()) // Contructor
    .def_readwrite("x_min", &BoundingBox::x_min)
    .def_readwrite("y_min", &BoundingBox::y_min)
    .def_readwrite("x_max", &BoundingBox::x_max)
    .def_readwrite("y_max", &BoundingBox::y_max)
    .def_readwrite("confidence", &BoundingBox::confidence);

    
    m.def("which_model", &which_model, "Function which will decide which model is used between YOLOv8s(0; Default) and DETR(1)");
    m.def("reset_model", &reset_model, "Resets the active model back to YOLO (0)"); // You Wheezo!
}
