<!-- import the image.webp -->

<p align="center">
  <img src="assets/chewbacca.webp" alt="Alt text" title="Optional title" width="300" />
</p>



## Installation

Create the conda environment `conda create -n chewbacca python=3.10`. xformers require python>=3.10

then install all other packages here:


```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
git clone git@github.com:brjathu/Chewbacca_test.git; cd Chewbacca_test
git checkout gsplat1.x
pip install -r requirements.txt
conda install xformers -c xformers
pip install -e .
```