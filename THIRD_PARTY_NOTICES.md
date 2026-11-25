# Third-Party Notices

`tiny-ja-wakeword` distributes quantized ONNX acoustic models trained via knowledge distillation using the following open datasets and open-source pretrained models:

1. **Japanese Phoneme Zipformer Teacher (`mitsu-h/icefall-ja-phoneme-kws-cv25`)**
   - Source: https://huggingface.co/mitsu-h/icefall-ja-phoneme-kws-cv25
   - License: Apache License 2.0
   - Used as a frozen teacher encoder during offline distillation. No original Zipformer weights are redistributed in this repository.

2. **Mozilla Common Voice Corpus 27.0 (Japanese)**
   - License: Creative Commons CC0 1.0 Universal (Public Domain Dedication)
   - Used for speech distillation and continuous evaluation benchmarks (strictly speaker- and phrase-disjoint splits).

3. **OpenSLR SLR28: Room Impulse Response and Noise Database (`RIRS_NOISES`)**
   - License: Apache License 2.0 (with `pointsource_noises` recordings marked as Public Domain on Freesound)
   - Reference: Tom Ko, Vijayaditya Peddinti, Daniel Povey, Michael L. Seltzer, and Sanjeev Khudanpur, *"A Study on Data Augmentation of Reverberant Speech for Robust Speech Recognition"*, ICASSP 2017.
   - Used for reverberation and environmental noise augmentation (strictly room- and source-disjoint train/dev/eval splits).

4. **Runtime Dependencies**
   - `onnxruntime` (MIT License)
   - `kaldi-native-fbank` (Apache License 2.0)
   - `numpy` (BSD 3-Clause License)
   - `soxr` (LGPL-2.1-or-later)
