# Related work

## Question checked

Whether any public source already provides the bounding box or pixel offset locating a BS-80K region crop inside its whole body ANT/POST source image, the thing this project's template matching is trying to recover. Checked by web search, session dated 2026-07-12.

## Official download, no extra annotation files

The Google Drive folder linked from the paper holds a single file, temp.zip, 697.2 MB, last modified 28 May 2022. No changelog, release notes, or extra annotation files are visible in the Drive listing itself.

https://drive.google.com/drive/folders/1DOBkLXgQeREQjF-nQIGNBBzPCb5s7RNu

## Unofficial mirror, no extra annotation files

Kaggle mirror by user mariusmarin, same 26 region folders plus wholeBodyANT/POST, same file counts, plus one readme.txt. No description published on the Kaggle page, no bounding box or offset files beyond the folder structure already known from the paper.

https://www.kaggle.com/datasets/mariusmarin/bs-80k

## Citations of the BS-80K paper, none address region crop location

15 papers cite the BS-80K paper (DOI 10.1016/j.compbiomed.2022.106221) per Semantic Scholar, checked 2026-07-12. All use BS-80K for hot spot or lesion detection, whole image classification, skeleton or hot spot segmentation, image enhancement, inpainting, federated learning, or PHI redaction benchmarking. None discuss or provide a bounding box or pixel offset locating a region crop inside its whole body source.

- Automated Detection of Pelvic Bone Metastases in Radionuclide Bone Scan Images Using a Dual-Stream Cross-Attention Transformer Model, https://doi.org/10.1109/QPAIN69676.2026.11546572
- WBSeg-Swin, Repurposing Swin Transformers for Bone Metastasis Detection in Whole-Body Scans, https://doi.org/10.1109/ACDSA67686.2026.11467597
- DBFANet, a dual-branch feature alignment network for automated detection of breast cancer bone metastasis, https://doi.org/10.1088/1361-6560/ae4166
- Automated bone metastasis detection in whole-body bone scan using a triplet attention mechanism, https://doi.org/10.1117/12.3089535
- An Image Enhancement Algorithm for Whole-Body Bone Scans Based on Improved Particle Swarm Optimisation, https://doi.org/10.1109/CISP-BMEI68103.2025.11259406
- Automatic Bone Metastasis Detection Using Yolov11, https://doi.org/10.1109/ICITACEE66165.2025.11232926
- Hotspot Segmentation in Whole-Body Bone Scans for Cancer Metastasis Detection Using U-Net++, https://doi.org/10.1109/ICoICT66265.2025.11192897
- Automatic detecting multiple bone metastases in breast cancer using deep learning based on low-resolution bone scan images, https://doi.org/10.1038/s41598-025-92594-5
- Generative artificial intelligence enables the generation of bone scintigraphy images and improves generalization of deep learning models in data-constrained environments, https://doi.org/10.1007/s00259-025-07091-8
- Exploring AI-Based System Design for Pixel-Level Protected Health Information Detection in Medical Images, https://doi.org/10.1007/s10278-025-01619-y
- AI-powered automated analysis of bone scans, a survey, https://doi.org/10.1049/ipr2.13311
- Artificial Intelligence in Bone Metastasis Analysis, Current Advancements, Opportunities and Challenges, https://doi.org/10.48550/arXiv.2404.19598
- A Federated Learning Approach to Bone Metastasis Prediction Using Convolutional Neural Network, https://doi.org/10.1109/ICCIT60459.2023.10441154
- A Novel Diffusion-Model-Based Bone Scan Image Inpainting Algorithm, https://doi.org/10.1109/BIBM58861.2023.10386039
- Artificial intelligence-based radiomics in bone tumors, technical advances and clinical application, https://doi.org/10.1016/j.semcancer.2023.07.003

## No dataset or repo found providing the region crop bounding box

No GitHub repo, Kaggle dataset, or Roboflow project was found providing a bounding box or pixel offset locating a BS-80K region crop inside its whole body source.

Roboflow user KneeJointLocalizationJan2023 runs several projects with a bs-80k prefix, bs-80k-test, bs-80k-few-shot, bs-80k-few-shot-k7, bs-80k-few-shot-k10. These label whole images Normal or Abnormal as an object detection task, they are not region crop localization, and do not use the region folders at all.

https://universe.roboflow.com/kneejointlocalizationjan2023

## No public implementation of the 2007 reference point algorithm found

No code implementation was found for the reference point and borderline segmentation described in Huang, Kao, Chen 2007, IEEE Trans. Nucl. Sci. 54(3), 514-522. Only the paper text itself was found, for example on ResearchGate.

https://www.researchgate.net/publication/3140428_A_Set_of_Image_Processing_Algorithms_for_Computer-Aided_Diagnosis_in_Nuclear_Medicine_Whole_Body_Bone_Scan_Images

Related but distinct methods exist for the general problem of segmenting a whole body bone scan into skeletal regions. These are not implementations of the 2007 algorithm and are not tied to BS-80K's specific crop pipeline, listed here only because they turned up in the same search.

- Automatic whole-body bone scan image segmentation based on constrained local model, landmark based, menpo library, https://www.researchgate.net/publication/346723479_Automatic_whole-body_bone_scan_image_segmentation_based_on_constrained_local_model
- Efficient-BtrflyNet, CNN based skeleton segmentation, evaluated on 37 images, a different dataset, https://link.springer.com/article/10.1007/s44196-024-00453-4

## Conclusion

Nothing public was found that already provides the bounding box or pixel offset locating a BS-80K region crop inside its whole body source image. The template matching approach in this repo is not made redundant by anything found in this search.
