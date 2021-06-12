This repository contains our code for AIC19 track3 at CVPRW19.

Requirements:
- [Bash shell](https://www.gnu.org/software/bash/)
- [Slurm](https://slurm.schedmd.com/overview.html)
- [Python 3](https://www.python.org/)
- [Caffe](https://caffe.berkeleyvision.org/)
- [OpenCV 3](https://opencv.org/)
- [Pytorch 0.4](https://pytorch.org/)

1. Run bg.py to produce background and foreground frames for each video, and save them in ./test_bg_imgs and
   ./test_fg_imgs respectively.

2. Run detection on all test videos in "./detection" directory.

3. Run all scripts in ./preprocess directory to make video masks, locate_stuck and filter unsuitable bboxes.

4. Run tracking codes to obtain the final results in './SOT'
