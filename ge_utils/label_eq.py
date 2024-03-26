import numpy as np


LABELS_SUPPORTED = [
    "obj_det_AP",
    "inst_seg_AP",
    "sem_seg_mIoU",
    "pan_seg_PQ",
    "flops_pan_seg",
    "flops_inst_seg",
    "flops_sem_seg",
    "flops_obj_det",
    "acc",
    "flops",
    "latency_gpu",
    "latency_cpu",
    "latency_npu",
    "n_params",
]


def process_label(entry, label_str="acc"):
    
    for label in LABELS_SUPPORTED:
        if label in label_str:
            if label in label_str:
                label_str = label_str.replace(label, str(entry[label]))
            else:
                return None

    return eval(label_str)


def format_target(label_str):
    label_str = label_str.replace("/", "_div_")
    label_str = label_str.replace(".", "")
    label_str = label_str.replace(" ", "")
    label_str = label_str.replace("(", "")
    label_str = label_str.replace(")", "")
    label_str = label_str.replace("[", "")
    label_str = label_str.replace("]", "")
    label_str = label_str.replace("'", "")
    label_str = label_str.replace("*", "")
    label_str = label_str.replace(">=", "_gte_")
    label_str = label_str.replace(">", "_gt_")
    label_str = label_str.replace("<=", "_lte_")
    label_str = label_str.replace("<", "_lt_")
    return label_str