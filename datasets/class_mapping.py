from PIL import Image
import numpy as np

mapped_class_names  = {
    0:  "Background",
    1:  "Tools/camera",
    2:  "Vein",
    3:  "Artery",  
    4:  "Nerve",
    5:  "Small intestine",
    6:  "Colon/rectum",
    7:  "Abdominal wall",
    8:  "Diaphragm",
    9:  "Fat",
    10: "Liver",
    11: "Bile/lymph Duct",
    12: "Gallbladder",
    13: "Hepatic ligament",
    14: "Cystic plate",
    15: "Stomach",
    16: "Spleen",
    17: "Uterus",
    18: "Ovary",
    19: "Oviduct",
    20: "Prostate",
    21: "Urethra",
    22: "Ligated plexus",
    23: "Seminal vesicles",
    24: "Non anatomical",
    25: "Bladder",
    26: "Lung",
    27: "Airway (bronchus/trachea)",
    28: "Esophagus",
    29: "Pericardium",
}

mapping = {
    0:  0,  # Background"
    1:  1,  # Tools/camera
    2:  2,  # Vein
    3:  3,  # Artery
    4:  4,  # Nerve
    5:  5,  # Small intestine
    6:  6,  # Colon/rectum
    7:  7,  # Abdominal wall
    8:  8,  # Diaphragm
    9:  9,  # Omentum
    10: 3,  # Aorta => Artery
    11: 2,  # Vena cava => Vein
    12: 10, # Liver
    13: 11, # Cystic duct => Bile/lymph Duct
    14: 12, # Gallbladder
    15: 2,  # Hepatic vein => Vein
    16: 13, # Hepatic ligament
    17: 14, # Cystic plate
    18: 15, # Stomach
    19: 11, # Ductus choledochus => Bile/lymph Duct
    20: 9,  # Mesenterium => Fat
    21: 11, # Ductus hepaticus => Bile/lymph Duct
    22: 16, # Spleen
    23: 17, # Uterus
    24: 18, # Ovary
    25: 19, # Oviduct
    26: 20, # Prostate
    27: 21, # Urethra
    28: 22, # Ligated plexus
    29: 23, # Seminal vesicles
    30: 24, # Catheter => Non anatomical
    31: 25, # Bladder
    32: 0,  # Kidney => Background
    33: 26, # Lung
    34: 27, # Airway (bronchus/trachea)
    35: 28, # Esophagus
    36: 29, # Pericardium
    37: 2,  # V azygos => Vein
    38: 11, # Thoracic duct => Bile/lymph Duct
    39: 4,  # Nerves => Nerve
    40: 0,  # Ureter => Background
    41: 24, # Non anatomical structures => Non anatomical
    42: 0,  # Excluded frames => Background
    43: 0,  # Mesocolon => Background
    44: 0,  # Adrenal Gland => Background
    45: 0,  # Pancreas => Background
    46: 0,  # Duodenum => Background
}

def remap_mask(mask: np.ndarray, mapping: dict, default_value=0) -> np.ndarray:
    """
    Remap class labels in a 2D segmentation mask using a lookup table (LUT).

    Pixel-wise labels in `mask` are converted according to `mapping` using
    fast NumPy indexing. Labels not present in `mapping` are set to
    `default_value`.

    Parameters
    ----------
    mask : np.ndarray
        2D array of shape (H, W) containing integer class labels.
    mapping : dict
        Mapping from original labels to new labels.
    default_value : int, optional
        Value assigned to unmapped labels (default: 0).

    Returns
    -------
    np.ndarray
        Remapped mask of shape (H, W).
    """

    if mask.ndim != 2:
        raise ValueError("Mask must be 2D (H, W)")

    mask = mask.astype(np.int32)

    # LUT size based on maximum possible label (consider both mapping keys and actual mask values)
    max_label = max(mask.max(), max(mapping.keys()))

    # Fill LUT with default value
    lut = np.full(max_label + 1, default_value, dtype=np.int32)

    # Assign mappings
    for k, v in mapping.items():
        lut[k] = v

    # Apply mapping (vectorized, extremely fast)
    return lut[mask]


if __name__ == "__main__":
    path = r"D:\important datasets\SurgeSAM_final_split\SurgeSAM_final_split\test\esophagectomy\6X7BRo4hNt8#si=4hwfn4vtn9UUEJgT_ROBOT\clip_0001\masks\frame_000140.png"

    # Load image palette image as greyscale image
    mask = Image.open(path)
    mask = np.array(mask)
    
    # Remap mask using the mapping dictionary
    remapped_mask = remap_mask(mask, mapping)

    print("Original unique labels:", np.unique(mask))
    print("Remapped unique labels:", np.unique(remapped_mask))