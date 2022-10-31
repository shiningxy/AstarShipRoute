# AstarShipRoute

## Install

### download GDAL

进入https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal

`Ctrl-F` 搜索 `GDAL‑3.3.3‑cp38‑cp38‑win_amd64.whl` 并下载

```
pip install GDAL‑3.3.3‑cp38‑cp38‑win_amd64.whl
```

### download ETOPO1_Bed_c_geotiff.tif

https://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/cell_registered/georeferenced_tiff/ETOPO1_Bed_c_geotiff.zip

保存至`./data/`

```
conda create -n asr python=3.8 -y
conda activate asr
pip install -r requirements.txt
```

