Code to train OpenAI CLIP model for image-2-image similarity check


# How to run CLIP training:
1. Download true pair dataset and .csv with filenames
2. Create conda environment
    ```.bash
    $ conda create --name=trainclip  python=3
    $ conda install --yes -c pytorch pytorch=1.13.1 torchvision cudatoolkit=11.0
    $ pip install git+https://github.com/openai/CLIP.git
    $ pip install pandas
    ```
3. Run `train.py`

conda list --explicit > spec-file.txt
conda create --name trainclip --file spec-file.txt
conda install --name trainclip --file spec-file.txt
`conda remove -n trainclip --all`