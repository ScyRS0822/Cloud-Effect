# Cloud Effect on Surface Temperature
The code and description related to my CET paper

The functions of these Python files are as follows:

1) downERA5: Batch download of ERA5 (including all relevant input parameters).
2) setDataset: Randomly sample the ERA5 and GLASS data downloaded for each year to generate training and testing sets. Note that we multiply LWDR and BBE before adding them to the sample set, as LWDR * BBE is more representative of the long-wave energy absorbed by the surface; Furthermore, day of year (DoY) is a positive number in the northern hemisphere and a negative number in the southern hemisphere, thus expressing the fact that the seasons in the northern and southern hemispheres are opposite. The above operations are conducive to better guiding machine learning models for learning.
3) Train: Train three machine learning models (Random Forest, CatBoost, LightGBM) using the sample set produced in the setDataset file.
4) Test: Utilize a new year to verify the accuracy of trained machine learning models (Fig. 2).
5) Curve: Generate the daily variation curves of surface temperature (Fig. 1 and Fig. 3).
6) Yearly0: Generate the global average annual CET (Fig. 4).
7) 3poles: Generate the Monthly Global Average CET image.
8) Excel: Using global monthly average CET images to calculate monthly average CET at global and 3-poles scales (Fig. 5).
