# equations

## Presentation

Final project at the Wagon's Data Science bootcamp realized with Hanna Chartier and Jocelyn Romero in May 2022.  
The aim of the project was to develop a solution to recognize handwritten math caracters and translate them in LaTeX thanks to Deep Learning.  
It would fasten the way a LaTeX document is written.

## Process

When an image is passed to our program, thanks to the OpenCV librairy , each symbol of the mathematical sentence is extracted one by one (we used contour recognition). Then, each element is fed to a Convolutionnal Neural Network that will determine which symbol it is. Finally, each recognized symbol is mapped into its LaTeX transcription.  
An API with a website were designed which could receive an image and translate it to LaTeX. It was also possible to draw our math symbols directly on the website.  

## Possible improvement

Even if it is trained on a big dataset avaible on Kaggle (https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols?select=data.rar), our CNN model can struggle to identify correctly the symbol. Maybe, some feature engineering could done or model tuning.  
Our program is capable to extract squared root, exponents and index thanks to the OpenCV part. However, some improvements can be made in order to process integrals and fractions for instance.  
