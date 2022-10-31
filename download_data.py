from turtle import down
import requests
import os
import zipfile


"""
    download  ETOPO1_Bed_c_geotiff.tif  to  ./data/
"""
def download(path):
    file_name = 'ETOPO1_Bed_c_geotiff.zip'
    folder = 'data'
    file_path = folder + "/" + file_name
    req = requests.get(f"https://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/cell_registered/georeferenced_tiff/{file_name}")
    with open(path, "wb") as file:
        file.write(req.content)
    unzip(file_path, folder)

"""
    unzip  ETOPO1_Bed_c_geotiff.zip
"""
def unzip(zip_src, unzip_path):
    is_zip = zipfile.is_zipfile(zip_src)
    if is_zip:
        zip_file = zipfile.ZipFile(zip_src, 'r')
        for file in zip_file.namelist():
            zip_file.extract(file, unzip_path)
        print("Unzip success!")
    else:
        print("This is not zip file!")

"""
    check  and  download  ETOPO1_Bed_c_geotiff.tif  exists  or  not
"""
def main():
    path = "data/ETOPO1_Bed_c_geotiff.tif"
    if os.path.exists(path):
        print(path, "exists!")
    else:
        download(path)
    
if __name__ == "__main__":
    main()