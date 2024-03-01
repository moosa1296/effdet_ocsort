import torch
from train import *
from ocsort_tracker.ocsort import OCSort
import motmetrics as mm
import numpy as np
from args import make_parser
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
def calculate_iou(box1, box2):
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def evaluate_detection_performance(predicted_bboxes, ground_truth, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth)

    for pred_bbox in predicted_bboxes:
        matched = False
        for gt_bbox in ground_truth:
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou >= iou_threshold:
                true_positives += 1
                false_negatives -= 1
                matched = True
                break
        if not matched:
            false_positives += 1

    precision = true_positives / max((true_positives + false_positives), 1)
    recall = true_positives / max((true_positives + false_negatives), 1)
    f1 = 2 * (precision * recall) / max((precision + recall), 1)
    return precision, recall, f1, true_positives, false_positives, false_negatives

def save_detection_results(detections, confidences, labels, frame_number, file_path):
    with open(file_path, 'a') as file:
        for bbox, score, label in zip(detections, confidences, labels):
            x_min, y_min, x_max, y_max = bbox
            formatted_result = f"{frame_number},{x_min},{y_min},{x_max},{y_max},{score},\n"
            file.write(formatted_result)

def save_tracking_results(all_tracking_results, results_path):
    sorted_tracking_data = sorted(all_tracking_results, key=lambda x: x[1])

    with open(results_path, 'w') as file:  
        for target, frame_number in sorted_tracking_data:
            x_min, y_min, x_max, y_max, track_id, cat_id, frame_lag = target
            formatted_results = f"{x_min},{y_min},{x_max},{y_max},{track_id},{cat_id},{frame_lag},{frame_number}\n"
            file.write(formatted_results)


def main(args):
    
    dataset_path = Path("/home/user-1/pig_dataset/val")
    summary = []
    for folders in os.listdir(dataset_path):
        val_images_path = dataset_path / folders / "images"
        val_anns_path = dataset_path / folders / "annotations"
        pigs_val_ds = PigsDatasetAdapter(val_images_path, val_anns_path)
        detection_results_path = f'/home/user-1/results/tracking_results_{folders}.txt'

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        all_tracking_results = []
        model = EfficientDetModel(
            num_classes=1,
            img_size=img_size[0],
            model_architecture=architecture,
            iou_threshold=0.5,
            prediction_confidence_threshold=0.45,
            sigma=0.8,
        )

        model.load_state_dict(
            torch.load("/home/user-1/results/effdet_tf_efficientdet_d0_(512, 512).pth")
        )
        model.eval()

        tracker = OCSort(det_thresh=0.4, iou_threshold=0.2)
        mot_accumulator = mm.MOTAccumulator(auto_id=True)

        
        for i in range(len(pigs_val_ds)):
        
            image, ground_truth, _, _, _ = pigs_val_ds.get_image_and_labels_by_idx(i)
            
            predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict([image])
            
            true_positives, false_positives, false_negatives = evaluate_detection_performance(predicted_bboxes[0], ground_truth)
            
            save_detection_results(predicted_bboxes[0], predicted_class_confidences[0], predicted_class_labels, i, detection_results_path)
        
            scores = np.array(predicted_class_confidences[0])
            detections = np.array(predicted_bboxes[0])
            categories = np.array(predicted_class_labels).flatten()

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            online_targets = tracker.update_public(detections, categories, scores)

            for target in online_targets:
                all_tracking_results.append((target, i)) 

            gt_ids = list(range(len(ground_truth)))
            gt_boxes = ground_truth.tolist()
            tracker_ids = [obj[4] for obj in online_targets]
            tracker_boxes = [obj[:4] for obj in online_targets]

            distances = mm.distances.iou_matrix(gt_boxes, tracker_boxes, max_iou=1 - 0.5)

            mot_accumulator.update(
                gt_ids,
                tracker_ids,
                distances
            )

        save_tracking_results(all_tracking_results, f'/home/user-1/results/tracking_results_{folders}.txt')
        
        overall_precision = total_true_positives / max((total_true_positives + total_false_positives), 1)
        overall_recall = total_true_positives / max((total_true_positives + total_false_negatives), 1)
        overall_f1 = 2 * (overall_precision * overall_recall) / max((overall_precision + overall_recall), 1)

        print(f"Overall Detection Performance: Precision={overall_precision}, Recall={overall_recall}, F1={overall_f1}")

        mh = mm.metrics.create()
        summary.append(mh.compute(mot_accumulator, metrics=mm.metrics.motchallenge_metrics, name='acc'))
        
        output_file_path = "/home/user-1/results/mot_metrics_summary2.txt"
        with open(output_file_path, 'w') as f:
            for idx, df in enumerate(summary):
                if len(summary) > 1:
                    f.write(f"File : {folders} \n Results for Accumulator {idx + 1}:\n")
                df_str = df.to_string(index=False)
                f.write(df_str)
                f.write("\n\n")
        print(f"Summary saved to {output_file_path}")
    mota = 0
    motp = 0
    idf1 = 0
    idp = 0
    idr = 0
    for sum in summary:
        mota += sum['mota'].iloc[0]
        motp += sum['motp'].iloc[0]
        idf1 += sum['idf1'].iloc[0] 
        idp += sum['idp'].iloc[0]  
        idr += sum['idr'].iloc[0]
    avg_mota = mota / len(summary) 
    avg_motp = motp / len(summary)
    avg_idf1 = idf1 / len(summary)
    avg_idp = idp / len(summary)
    avg_idr = idr / len(summary)

    print(f"IDR: {avg_idr}\n IDF1: {avg_idf1}\n IDP: {avg_idp}\n MOTA: {avg_mota}\n MOTP: {avg_motp}")
    print(f"True Positives: {total_true_positives}\n False Positives: {total_false_positives}\n False Negatives: {total_false_negatives}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)





