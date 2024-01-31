## Ship Detection

#### Abstract

Synthetic Aperture Radar (SAR) images have advantage of being able to observe Earth surface regardless of wether condition and day and night. However, when applying ship detection techniques which is previously presented methods to inshore area, there is a limitation of having a low ship detection rate and a high false alarm rate due to artificial structures. In this study, research was conducted to improve ship detection performance using the statistical and deep learning methods presented in previous studies. In the statistical method, a Frozen Background Reference (FBR) image was generated using amplitude variance, and Constant False Alarm Rate (CFAR) algorithm was applied to the ratio image generated as an FBR image. CFAR-guided Convolutional Neural Network (CG-CNN) was performed to apply deeplearning method to detect ships in inshore area. 

#### Ship detection data download link
Sentinel-1A Images: https://drive.google.com/file/d/1tiJ6RUPiBRDZyD8VR5iPQmCbGe0IyVhZ/view?usp=sharing

Deep learning training data: https://drive.google.com/file/d/1my-He41in0sf6ddqZZXK9t5lPuU8pO_F/view?usp=sharing

Deep learning label data: https://drive.google.com/file/d/1gGRlTcSlLpuhdNz9dIM5Fl4STMFqGv-P/view?usp=sharing

### About OriBuri.py

OriBuri is a SAR(Synthetic Aperture Radar) Image Processing & Application Software based on SIPAS which is
developed by GEO-CODING, a study group of ungraduated students of Sejong University in Republic
of Korea, department of Earth resource engineering. (https://sites.google.com/view/sejong-geocoding/remote-sensing). OriBuri is mainly focused on
Sentinel-1 product and KOMPSAT-5(Korea Multi-Purpose Satellite-5) which is developed by KARI(Korea Aerospace Research Institute, https://www.kari.re.kr/eng/sub03_03_01.do). With OriBuri,
you can easily handle SAR data in python.

## Reference

Thibault Taillade, Laetitia Thirion-Lefevre and Régis Guinvarc’h, 2020.
Detecting Ephemeral Objects in SAR Time-Series Using Frozen Background-Based Change Detection,
MDPI, Remote Sens. 2020, 12(11), 1720; https://doi.org/10.3390/rs12111720

Shao, .Z et al, 2023.
CFAR-guided Convolutional Neural Network for Large Scale Scene SAR Ship Detection,
IEEE, Radar Conference, DOI:10.1109/RADARCONF2351548.2023.10149747

Chen, .Z et al, 2023.
Inshore Ship Detetion Based on Multi-Modality Saliency for Synthetic Aperture Radar Images,
MDPI, Remote Sens. 2023, 15, 3868. https://doi.org/10.3390/rs15153868
