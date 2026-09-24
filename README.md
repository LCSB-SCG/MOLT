# MOLT: Multi-Object and Lineage Tracking in 2D and 3D Biomedical Time-Series Imaging

MOLT is a robust multi-object and lineage tracking framework designed to handle challenging scenarios in 2D and 3D volumetric biomedical imaging, such as camera misalignment, deformable tissues, and morphological changing instances.
It combines affine and deformable registration techniques with an overlap-based object and lineage tracking algorithm, enabling consistent identification of physical objects over time.

The code is split into multiple modules, each consisting of one step of the pipeline.

## Running the code:

- To run the code you can install the Docker Dev Containers extension in VSCode (ms-vscode-remote.remote-containers) or you can create a docker container using the following command:

        docker build --tag 'molt-track  .

- Adjust the config file (assets/config_celltracking_template.json) to match your specific requirements (specific dataset, path,...)

- Run the following command in the container to run the pipeline for the specific dataset:

        python -m src/main.py --config="<PATH/TO/YOUR/CONFIG_FILE.json>"


## Datasets

The cell tracking datasets can be automatically downloaded and processed using the CellTrackingDataset or under the following links:

- Fluo-N2DH-GOWT1: https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

- Fluo-N2DL-HeLa: https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip

- PhC-C2DL-PSC: https://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip


## Cell Tracking Results
Fluo-N2DH-GOWT1 Lineage and Individual Tracking Result
| | | | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
Region 1 | <img width="300" alt="Fluo-N2DH-GOWT1-lineage-01.gif" src=visuals/Fluo-N2DH-GOWT1-lineage-01.gif>   |  <img width="300" alt="Fluo-N2DH-GOWT1-tracked-01.gif" src=visuals/Fluo-N2DH-GOWT1-tracked-01.gif>|  Region 2 |<img width="300" alt="Fluo-N2DH-GOWT1-lineage-02.gif" src=visuals/Fluo-N2DH-GOWT1-lineage-02.gif>   |  <img width="300" alt="Fluo-N2DH-GOWT1-tracked-02.gif" src=visuals/Fluo-N2DH-GOWT1-tracked-02.gif>

Fluo-N2DL-HeLa Lineage and Individual Tracking Result
| | | | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
Region 1 | <img width="300" alt="Fluo-N2DL-HeLa-lineage-01.gif" src=visuals/Fluo-N2DL-HeLa-lineage-01.gif>   |  <img width="300" alt="Fluo-N2DL-HeLa-tracked-01.gif" src=visuals/Fluo-N2DL-HeLa-tracked-01.gif>|  Region 2 |<img width="300" alt="Fluo-N2DL-HeLa-lineage-02.gif" src=visuals/Fluo-N2DL-HeLa-lineage-02.gif>   |  <img width="300" alt="Fluo-N2DL-HeLa-tracked-02.gif" src=visuals/Fluo-N2DL-HeLa-tracked-02.gif>

PhC-C2DL-PSC Lineage and Individual Tracking Result
| | | | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
Region 1 | <img width="300" alt="PhC-C2DL-PSC-lineage-01.gif" src=visuals/PhC-C2DL-PSC-lineage-01.gif>   |  <img width="300" alt="PhC-C2DL-PSC-tracked-01.gif" src=visuals/PhC-C2DL-PSC-tracked-01.gif>|  Region 2 |<img width="300" alt="PhC-C2DL-PSC-lineage-02.gif" src=visuals/PhC-C2DL-PSC-lineage-02.gif>   |  <img width="300" alt="PhC-C2DL-PSC-tracked-02.gif" src=visuals/PhC-C2DL-PSC-tracked-02.gif>

## Publication and Citation
The paper related to this work can be found [placeholder until published](https://link.springer.com/article/10.1186/s12859-026-06434-y). Please cite this work the following publication is used in your research.
