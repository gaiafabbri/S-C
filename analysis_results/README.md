## Analysis.py
This code implements some function to analyse and visualize the input data; several functions are implemented:
- _plot_images(num_images_to_plot, signal_data, background_data)_: takes as argument the number of images that the users wants to plot, the signal and background data; it is useful to visualize some images of the dataset, comparing signal and background
- _pixel_intensity_distribution (sgn_mean,bkg_mean)_: it takes as arguments two arrays containing the mean pixel intensities; this value are used to obtrain the histograms of the intensity distribution. It is useful to understand how pixel are distributed within the images and to look for differences among signal and background data
- _plot_cluster_histogram(cluster1, cluster2)_: it shows a histogram of clusters coming from the KMeans clustering algorithm; it takes as arguments two arrays containign the label of the cluster for each image. It is useful to look for similarities within the data that creates some substractures, called clusters; within clusters, data are considered homogeneous according to a metric defined by the clustering algorithm, in this case the euclidean distance amogn points. In particular, it is helpful to distinguish if the two classes are grouped differently and to simplify data analysis and understanding
- _plot_cluster_centers(centers1, centers2)_: it takes as arguments the centroids for signal and background clusters, obtained by the clustering algorithm as the mean representation of points within a cluster; it is helpful to focus on the principal features of data
- _plot_pixel_distribution(signal_image, background_image)_: this function shows the distribution (the histogram) of pixel within the classes, together with the pixel correltation; it looks for pixel correlation and helps to understand differences in the intensity of images
- _plot_intensity_profile(image_data1,image_data2, axis='row')_: it takes as argument the signal and background arrays; it is the visualization of the intensity profile of images along rows or columns, looking for differnces between signal and background alogn different directions

__how to run:__ in the main folder: 

$ python3 Analysis.py

The resulting plots are saved in the "__analysis_results__" folder and show no significant differences between signal and background events; the pixel distribution and the pixel intensity distribution have a comparable behaviour, with some differences due to the intrinsic nature of the data. The same can be observed for the intensity profile and the cluster histograms: the two classes are distinguishable, but there is no bias in the distributions that could have affected the training, resulting in an overly simple classification.
