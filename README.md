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
v1.0: 
compare cca, ecca, etrca, ms-ecca, ms-etrca (16 May 2020)

v1.1: 
add notch filter
modify the bandpass filter design
add ms-ecca+ms-etrca and TDCA for comparison (25 Nov 2021)

# Result
...

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
 
 @article{wong2020spatial,
 
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
