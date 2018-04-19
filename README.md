# MasterThesis
Deep learning approaches for retinal blood vessels segmentation

CNN - as a segmentation approach:
  - U-net Implementation (Based on MIT´s https://github.com/orobix/retina-unet).
  - Liskowski et al [1] No-pool Implementation. 
  - Liskowski et al [1] 3x3 No-pool Structure Prediction.
  - Classic CNN Structure.
  
CNN - as a feature extractor:
  - Wang et al [2] as feature extractor and Random Forest Classifier & diferent emsemble methods.
  - Wang et al [2] as feature extractor and SVM Classifier & diferent emsemble methods.

Datasets:
- DRIVE [3] and STARE [4] Raw Datasets.
- DRIVE [3] and STARE [4] Pre processed Dataset:
  - Histogram normalization.
  - Gamma Corrections
  - Global Contrast Normalization.
  - Others Techniques
- DRIVE [3] and STARE [4] Augmented Dataset:
  - ZCA Whitening.
  - Rotation by an angle from [0-180º]. 
  - Flipping horizontally or vertically.
  - Gamma correction of Saturation and Value.  
  
  
  Contributors:
  - Ricardo Araujo (INESC - TEC), Msc
  - MIT´s https://github.com/orobix/retina-unet
  
  [1] - Liskowski P, Krawiec K. Segmenting retinal blood vessels with deep neural networks [J]. IEEE Trans Med Imaging. 2016;35(11):2369–80.
  [2] - Wang S, Yin Y, Cao G, et al. Hierarchical retinal blood vessel segmentation based on feature and ensemble learning [J]. Neurocomputing. 2015;149:708–17.
  [3] - https://www.isi.uu.nl/Research/Databases/DRIVE/
  [4] - http://cecas.clemson.edu/~ahoover/stare/
