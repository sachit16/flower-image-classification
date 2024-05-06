# Image-Classification-Streamlit-TensorFlow
A basic web-app for image classification using Streamlit,Flask and TensorFlow.

It classifies the given image of a flower into one of the following five categories :-  

1. astilbe
2. bellflower
3. black_eyed_susan
4. calendula
5. california_poppy
6. carnation
7. common_daisy
8. coreopsis
9. daffodil
10. dandelion
11. iris
12. magnolia
13. rose
14. sunflower
15. tulip
16. water_lily
## Architecture :
<img src ='Densenet121.png' width = 700>

## Links:
### 1.Dataset Link:-
* For 5 Flower Classes:-
** https://www.kaggle.com/datasets/imsparsh/flowers-dataset
* For 16 Flower Classes:-
** https://www.kaggle.com/datasets/l3llff/flowers
### 2.Model File:-
* https://drive.google.com/file/d/1e7QV6NQ9gBPTAp-gByaNH2RX34QT9Edt/view?usp=sharing
## Commands

To run the app locally, use the following command :-  
`streamlit run app_Streamlit.py`
Or
`python  app_Flask.py` 

The webpage should open in the browser automatically.  
If it doesn't, the local URL would be output in the terminal, just copy it and open it in the browser manually.  
By default, it would be `http://localhost:8501/`    For Streamlit
By default, it would be `http://localhost:5000/`    For Flask
Click on `Browse files` and choose an image from your computer to upload.  
Once uploaded, the model will perform inference and the output will be displayed.  

## Output
![Video](https://github.com/sachit16/flower-image-classification/blob/main/YouCut_20240505_234457700.mp4)

<img src ='/app/misc/Flask_home1.png' width = 700>
<img src ='/app/misc/Flask_home2.png' width = 700>

<img src ='/app/misc/sample_home_page.png' width = 700>  

<img src ='/app/misc/sample_output.png' width = 700>


## Notes
* A simple flower classification model was trained using TensorFlow.  
* The weights are stored as `flower_model_trained.hdf5`.  
* The code to train the modify and train the model can be found in `model.py`.  
* The web-app created using Streamlit can be found in `app_streamlit.py`
* The web-app created using Flask can be found in `app_Flask.py`


## References

* https://www.tensorflow.org/tutorials/images/classification
* https://docs.streamlit.io/en/stable/
