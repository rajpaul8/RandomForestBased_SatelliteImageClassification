{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Khushboo_RF.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyPYNhvPRKrnM8E3aDJGi+kn",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rajpaul8/SatelliteImageClassification/blob/main/RF_Classifier_Using_csv.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "x4TgPCDs49EW"
      },
      "source": [
        "IMPORTING MODULES"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-JcXliWb4qpx"
      },
      "source": [
        "import os\n",
        "import glob\n",
        "import matplotlib.pyplot as plt\n",
        "import rasterio as rio\n",
        "from rasterio.plot import plotting_extent\n",
        "import geopandas as gpd\n",
        "import earthpy as et\n",
        "import earthpy.spatial as es\n",
        "import earthpy.plot as ep\n",
        "import numpy as np\n",
        "import gdal\n",
        "import pandas as pd\n",
        "from sklearn.ensemble import RandomForestClassifier"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vSD1vKMv_7TY"
      },
      "source": [
        "Fetching The Imageries From the Path"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "s4VBEQa-45x4"
      },
      "source": [
        "#Input Dir\n",
        "path_of_unstackedBands = r\"/content/Khushboo/DelhiDataset\"\n",
        "search_Tif_With_Init = \"B*.tif\"\n",
        "FetchRasterBands = os.path.join(path_of_unstackedBands,search_Tif_With_Init)\n",
        "getListOfFetchedBands = glob.glob(FetchRasterBands)\n",
        "#print(getListOfFetchedBands)\n",
        "getListOfFetchedBands.sort()\n",
        "\n",
        "#Write Output Here:\n",
        "output_dir = \"/content/Khushboo/Output\"\n",
        "raster_out_path = os.path.join(output_dir, \"Stacked_Raster.tif\")\n",
        "\n"
      ],
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g9mTyVI848QP"
      },
      "source": [
        "Stacking Multiple Imageries into One Image with Multibands"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 592
        },
        "id": "s9hidbGu9UJ0",
        "outputId": "4751ea4c-9815-4880-970b-924fae475f91"
      },
      "source": [
        "array, raster_prof = es.stack(getListOfFetchedBands, out_path=raster_out_path)\n",
        "#Create Extent\n",
        "extent = plotting_extent(array[0], raster_prof[\"transform\"])\n",
        "\n",
        "#Plot and See\n",
        "fig, ax = plt.subplots(figsize=(12, 12))\n",
        "ep.plot_rgb(\n",
        "    array,\n",
        "    ax=ax,\n",
        "    stretch=True,\n",
        "    extent=extent,\n",
        "    str_clip=0.5,\n",
        "    title=\"Indices Stacked Image of First 3 Bands as RGB\",\n",
        ")\n",
        "plt.show()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 864x864 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "W4KAOA9rfS3W"
      },
      "source": [
        "# Removing No Data\n",
        "array_nodata, raster_prof_nodata = es.stack(getListOfFetchedBands, nodata=-9999)\n",
        "\"\"\" View hist of data with nodata values removed\n",
        "ep.hist(\n",
        "    array_nodata,\n",
        "    title=[\n",
        "        \"Band 1 - No Data Values Removed\",\n",
        "        \"Band 2 - No Data Values Removed\",\n",
        "        \"Band 3 - No Data Values Removed\",\n",
        "        \"Band 4 - No Data Values Removed\", \n",
        "    ],\n",
        ")\n",
        "plt.show()\n",
        "\"\"\"\n",
        "# Recreate extent object for the No Data array\n",
        "\n",
        "extent_nodata = plotting_extent(\n",
        "    array_nodata[0], raster_prof_nodata[\"transform\"]\n",
        ")"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Gk82WtkWot63"
      },
      "source": [
        "RF_Classifier"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SXyU92CUoova",
        "outputId": "92e1ba8e-39df-4850-85bd-393ec04f0a3e"
      },
      "source": [
        "#Define The Input and Output Path\n",
        "\n",
        "inpRaster = '/content/Khushboo/Output/Stacked_Raster.tif' \n",
        "\n",
        "outClassifiedRaster = '/content/Khushboo/Output/RF_Classified_2.tif'\n",
        "\n",
        "#Read Training Sample csv file\n",
        "df = pd.read_csv('/content/Khushboo/TrainingSample/SelectedPointRasterBandStats.csv', delimiter=',')\n",
        "# To see the Column name for inputing the band name reference below use -- > df.head()\n",
        "#enter training data bands according to your csv columns name\n",
        "data = df[['Band1','Band2','Band3','Band4']]\n",
        "#enter training label according to your csv column name\n",
        "label = df['Decision']\n",
        "del df\n",
        "\n",
        "#No Changes Required Beyond This Point ----\n",
        "\n",
        "#open raster\n",
        "ds = gdal.Open(inpRaster, gdal.GA_ReadOnly)\n",
        "\n",
        "#get raster info\n",
        "rows = ds.RasterYSize\n",
        "cols = ds.RasterXSize\n",
        "bands = ds.RasterCount\n",
        "geo_transform = ds.GetGeoTransform()\n",
        "projection = ds.GetProjectionRef()\n",
        "\n",
        "#read as array\n",
        "array = ds.ReadAsArray()\n",
        "\n",
        "ds = None\n",
        "\n",
        "# #modify structure\n",
        "array = np.stack(array,axis=2)\n",
        "array = np.reshape(array, [rows*cols,bands])\n",
        "test = pd.DataFrame(array, dtype='float32')\n",
        "# del array\n",
        "\n",
        "# #set classifier parameters and train classifier --> Using 50 Trees and n_jobs = -1 => All the processors to make prediction in higher efficiency \n",
        "clf = RandomForestClassifier(n_estimators=50,n_jobs=-1,oob_score=True)\n",
        "clf.fit(data,label)\n",
        "del data\n",
        "del label\n",
        "print('Our OOB prediction of accuracy is: {oob}%'.format(oob=clf.oob_score_ * 100))\n",
        "bands = [1, 2, 3, 4]\n",
        "\n",
        "for b, imp in zip(bands, clf.feature_importances_):\n",
        "    print('Band {b} importance: {imp}'.format(b=b, imp=imp))\n",
        "\n",
        "# #predict classes\n",
        "y_pred = clf.predict(test)\n",
        "del test\n",
        "classification = y_pred.reshape((rows,cols))\n",
        "del y_pred\n",
        "\n",
        "def createGeotiff(outRaster, data, geo_transform, projection):\n",
        "    # Create a GeoTIFF file with the given data\n",
        "    driver = gdal.GetDriverByName('GTiff')\n",
        "    rows, cols = data.shape\n",
        "    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Int32)\n",
        "    rasterDS.SetGeoTransform(geo_transform)\n",
        "    rasterDS.SetProjection(projection)\n",
        "    band = rasterDS.GetRasterBand(1)\n",
        "    band.WriteArray(data)\n",
        "    rasterDS = None\n",
        "\n",
        "\n",
        "#export classified image\n",
        "createGeotiff(outClassifiedRaster,classification,geo_transform,projection)\n"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Our OOB prediction of accuracy is: 0.0%\n",
            "Band 1 importance: 0.28472222222222215\n",
            "Band 2 importance: 0.23472222222222228\n",
            "Band 3 importance: 0.17500000000000002\n",
            "Band 4 importance: 0.3055555555555555\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}