This repository can be used to convert the KITTI RAW dataset into:
1) KITTI 3D sequence dataset
2) KITTI RAW 3D sequence dataset

To create these datasets:
1) Download the KITTI RAW dataset using <a href="http://www.cvlibs.net/download.php?file=raw_data_downloader.zip" target="_blank">the raw dataset download script</a> of Omid Hosseini.
2) Use [RAW2KITTI.py](RAW2KITTI.py) to generate either the KITTI 3D sequence or the KITTI RAW 3D sequence dataset.

One can use [find_duplicates.py](find_duplicates.py) to find the duplicates between the KITTI 3D object detection dataset and the KITTI RAW dataset.






This repository contains code from PCDet and the 'parse XML files' from Christian Herdtweck.
```
When using this dataset in your research, plsease cite this dataset:
@inproceedings{Sluis2020,
  author = {van der Sluis, Joram R.},
  title = {3D Object Detection For Intelligent Vehicles},
  booktitle = {TU Delft education repository},
  year = {2020}
}
```

```
When using this dataset in your research, please cite the original KITTI RAW dataset:
@ARTICLE{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
}
```
