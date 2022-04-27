# SSVEP-MSCCA-MSTRCA

This demo shows an example of using multi-stimulus extended canonical correlation analysis (ms-eCCA) and multi-stimulus ensemble task-related component analysis (ms-eTRCA) SSVEP recognition. Besides, the extended CCA (eCCA), the traditional eTRCA, and the task discriminant component analysis (TDCA) are also provided for comparison study.

For the above algorithms, please refer the following papers for more details:

ms-eCCA: Wong, C. M., et al. (2019). Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs. Journal of neural engineering.

ms-eTRCA: Wong, C. M., et al. (2019). Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs. Journal of neural engineering.

eCCA: Chen, X., et al. (2015). High-speed spelling with a noninvasive brain-computer interface. Proceedings of the national academy of sciences, 112(44), E6058-E6067.

eTRCA: Nakanishi, M., et al. (2017). Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis. IEEE Transactions on Biomedical Engineering, 65(1), 104-112.

TDCA: Liu, B., et al. (2021). Improving the Performance of Individually Calibrated SSVEP-BCI by Task-Discriminant Component Analysis. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29, 1998-2007.

In this example, most parameters (such as number of harmonics, time-window lengths and number of neighboring templates) can be adjusted manually to explore their effects on the final performance

This code is prepared by Chi Man Wong (chiman465@gmail.com)

# Version 
v1.0: (16 May 2020)<br>
compare cca, ecca, etrca, ms-ecca, ms-etrca <br>

v1.1: (25 Nov 2021)<br>
add notch filter<br>
modify the bandpass filter design<br>
add ms-ecca+ms-etrca and TDCA for comparison <br>
add if-else-end for selecting one specified dataset (tsinghua benchmark dataset or BETA dataset) <br>

v1.2: (26 Apr 2022)<br>
The same data length in the pre-processing, training, and testing <br>

# Preliminary results
Parameter setting:  
1) dataset_no=1; is_center_std=0; min_length=0.3; max_length=1.0; enable_bit=[1 1 1 1 1];  

We use 'plot_results.m' to plot the results.

![result1a](https://github.com/edwin465/SSVEP-MSCCA-MSTRCA/blob/master/result1a.png)

From the above figure (x-axis: data length, y-axis: accuracy), it seems that the TDCA achieves the highest performance (especially with the short data length). Meanwhile, the TDCA and the ms-eCCA+ms-eTRCA achieve similar performance, except Tw=0.3 and Tw=0.5.

![result1b](https://github.com/edwin465/SSVEP-MSCCA-MSTRCA/blob/master/result1b_.png)

2) dataset_no=2; is_center_std=0; min_length=0.3; max_length=1.0; enable_bit=[1 1 1 1 1];  

![result2](https://github.com/edwin465/SSVEP-MSCCA-MSTRCA/blob/master/result2.png)

From the above figure (x-axis: data length, y-axis: accuracy), it seems that the ms-eCCA+ms-eTRCA achieves the highest performance in most cases. Then the ms-eCCA looks also a bit better than the TDCA. Maybe the ms-eCCA+ms-eTRCA, ms-eCCA, and the TDCA provide the similar performance in statistical. Maybe later we will test them using paired t-test.  

According to the existing results, it can be found that in different datasets, the best algorithm is different. No algorithm can always perform the best.

# Citation
If you use this code for a publication, please cite the following papers

@article{wong2020learning,  
   title={Learning across multi-stimulus enhances target recognition methods in SSVEP-based BCIs},  
   author={Wong, Chi Man and Wan, Feng and Wang, Boyu and Wang, Ze and Nan, Wenya and Lao, Ka Fai and Mak, Peng Un and Vai, Mang I and Rosa, Agostinho},  
   journal={Journal of Neural Engineering},  
   volume={17},  
   number={1},  
   pages={016026},  
   year={2020},  
   publisher={IOP Publishing}  
 }  
 
 @article{wong2020spatial,<br> 
  title={Spatial filtering in SSVEP-based BCIs: unified framework and new improvements},  
  author={Wong, Chi Man and Wang, Boyu and Wang, Ze and Lao, Ka Fai and Rosa, Agostinho and Wan, Feng},  
  journal={IEEE Transactions on Biomedical Engineering},  
  volume={67},  
  number={11},  
  pages={3057--3072},  
  year={2020},  
  publisher={IEEE}   
}  

# Feedback
If you find any mistakes, please let me know via chiman465@gmail.com.
