# PALM-ASTRO
Palm-Astro History (Indian Data) Computer Vision + Multimodal Inference 

Dataset source and preprocessing summary 

data/
├── images/   # raw palm photos (.jpg/.png)
└── masks/    # corresponding mask images (.png, values 0–3)


Model architecture and training approach


The Palm Astro system uses a two-stage pipeline:
Palm Line Segmentation → U-Net (EfficientNet backbone)
Personality Classification → Feature-based ML model + Rule-based logic


Evaluation metrics (IoU, F1) 

(Example values — replace with your actual output from results/metrics/segmentation_metrics.json)

Line Type	IoU	Dice (F1)
Life Line	0.78	0.86
Head Line	0.71	0.82
Heart Line	0.67	0.75
Overall Mean	0.74	0.81

The exact metrics are saved in:

results/metrics/segmentation_metrics.json

Ethical/data privacy notes 

Data Privacy & User Protection
No Personal Identification

Palm Astro does not use palm images to identify individuals.

No attempts are made to infer identity, age, gender, or race from biometric patterns.

No Storage Without Consent

Uploaded images are processed in-memory during analysis.

Images are not saved, sent to remote servers, or reused without permission.

Anonymized Dataset

All dataset samples must avoid containing:

Faces

Background personal objects

Metadata with names, locations, or device details

Masks must contain only line annotations (no personal info).


Commands to run the app 

Install Dependencies
pip install -r requirements.txt

Train the Segmentation Model
python train_segmentation.py

Train the Personality Classifier
python train_classifier.py

Run Evaluation
python evaluate_segmentation.py

Classifier Evaluation:
python evaluate_classifier.py

Run the Palm Astro App (Streamlit)
streamlit run app.py

 Example output screenshots

 <img width="1917" height="917" alt="image" src="https://github.com/user-attachments/assets/e448432a-0924-4da4-81b3-51189a382e5a" />

 <img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/eef335ee-19ba-42c4-b2f8-8756295fcea1" />

