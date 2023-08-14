# CloudRadiativeEffect
The detail outputs related to my CET paper

The functions of these Python files are as follows:
1) setDataset: Randomly sample the ERA5 and GLASS data downloaded for each year to generate training and testing sets. Note that we multiply LWDR and BBE before adding them to the sample set, as LWDR * BBE is more representative of the long-wave energy absorbed by the surface; Furthermore, day of year (DoY) is a positive number in the northern hemisphere and a negative number in the southern hemisphere, thus expressing the fact that the seasons in the northern and southern hemispheres are opposite. The above operations are conducive to better guiding machine learning models for learning.
2) Train: Train three machine learning models (Random Forest, CatBoost, LightGBM) using the sample set produced in the setDataset file.
3) Test: Utilize a new year to verify the accuracy of trained machine learning models (Fig. 2).
4) Curve: Generate the daily variation curves of surface temperature (Fig. 1 and Fig. 3).
5) Yearly0: Generate the global average annual CET (Fig. 4).
6) 3poles: Generate the Monthly Global Average CET image.
7) Excel: Using global monthly average CET images to calculate monthly average CET at global and bipolar scales.
