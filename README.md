## Ship Detection

#### Abstract

Synthetic Aperture Radar (SAR) images have advantage of being able to observe Earth surface regardless of wether condition and day and night. However, when applying ship detection techniques which is previously presented methods to inshore area, there is a limitation of having a low ship detection rate and a high false alarm rate due to artificial structures. In this study, research was conducted to improve ship detection performance using the statistical and deep learning methods presented in previous studies. In the statistical method, a Frozen Background Reference (FBR) image was generated using amplitude variance, and Constant False Alarm Rate (CFAR) algorithm was applied to the ratio image generated as an FBR image. CFAR-guided Convolutional Neural Network (CG-CNN) was performed to apply deeplearning method to detect ships in inshore area. 

#### Ship detection data download link
Sentinel-1A Images: https://drive.google.com/file/d/1tiJ6RUPiBRDZyD8VR5iPQmCbGe0IyVhZ/view?usp=sharing

Deep learning training data: https://drive.google.com/file/d/1my-He41in0sf6ddqZZXK9t5lPuU8pO_F/view?usp=sharing

Deep learning label data: https://drive.google.com/file/d/1gGRlTcSlLpuhdNz9dIM5Fl4STMFqGv-P/view?usp=sharing

## OriBuri.py

## Reference

Thibault Taillade, Laetitia Thirion-Lefevre and Régis Guinvarc’h, 2020.
Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based Change Detection,
MDPI, Remote Sens. 2020, 12(11), 1720; https://doi.org/10.3390/rs12111720

