# IPL-Score-Prediction-using-Deep-Learning
 
Since the dawn of the IPL in 2008, it has attracted viewers all around the globe. A high level of uncertainty and last-minute nail-biters have urged fans to watch the matches. Within a short period, the IPL has become the highest revenue-generating league in cricket. In a cricket match, we often see the scoreline showing the probability of the team winning based on the current match situation. This prediction is usually done with the help of data analytics. Before, when there were no advancements in machine learning, predictions were usually based on intuition or some basic algorithms. The above picture clearly tells you how bad it is to take run rate as a single factor to predict the final score in a limited-overs cricket match.

predict-300x233
IPL Score Prediction

Being a cricket fan, visualizing the statistics of cricket is mesmerizing. We went through various blogs and found out patterns that could be used for predicting the score of IPL matches beforehand. 

Why Deep Learning?
We humans can’t easily identify patterns from huge data and thus here, machine learning and deep learning comes into play. It learns how the players and teams have performed against the opposite team previously and trains the model accordingly. Using only machine learning algorithm gives a moderate accuracy therefore we used deep learning which gives much better performance than our previous model and considers the attributes which can give accurate results.

Tools used:
Jupyter Notebook / Google colab
Visual Studio
Technology used:
Machine Learning.
Deep Learning
Well, for the smooth running of the project we’ve used few libraries like NumPy, Pandas, Scikit-learn, TensorFlow, and Matplotlib.
Step-by-step implementation:
First, let’s import all the necessary libraries:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import keras 
import tensorflow as tf
# Step 1: Loading the dataset!
When dealing with cricket data, it contains data from the year 2008 to 2017. The dataset can be downloaded from here. The dataset contain features like venue, date, batting and bowling team, names of batsman and bowler, wickets and more.

We imported both the datasets using .read_csv() method into a dataframe using pandas and displayed the first 5 rows of each dataset.

ipl = pd.read_csv('ipl_dataset.csv')
ipl.head()
Output:

    mid        date                  venue               bat_team  \
0    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
1    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
2    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
3    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
4    1  2008-04-18  M Chinnaswamy Stadium  Kolkata Knight Riders   
                     bowl_team      batsman   bowler  runs  wickets  overs  \
0  Royal Challengers Bangalore   SC Ganguly  P Kumar     1        0    0.1   
1  Royal Challengers Bangalore  BB McCullum  P Kumar     1        0    0.2   
2  Royal Challengers Bangalore  BB McCullum  P Kumar     2        0    0.2   
3  Royal Challengers Bangalore  BB McCullum  P Kumar     2        0    0.3   
4  Royal Challengers Bangalore  BB McCullum  P Kumar     2        0    0.4   
   runs_last_5  wickets_last_5  striker  non-striker  total  
0            1               0        0            0    222  
1            1               0        0            0    222  
2            2               0        0            0    222  
3            2               0        0            0    222  
4            2               0        0            0    222  

Step 3: Data Pre-processing
Dropping unimportant features

We have created a new dataframe by dropping several columns from the original DataFrame.
The new DataFrame contains the remaining columns that we are going to train the predictive model.

#Dropping certain features 
df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5','mid', 'striker', 'non-striker'], axis =1)
Further Pre-Processing

We have split the data frame into input features (X) and target features (y). Our target features is the total score.

X = df.drop(['total'], axis =1)
y = df['total']
Label Encoding

We have applied label encoding to your categorical features in X.
We have created separate LabelEncoder objects for each categorical feature and encoded their values.
We have created mappings to convert the encoded labels back to their original values, which can be helpful for interpreting the results.
#Label Encoding
 
from sklearn.preprocessing import LabelEncoder
 
# Create a LabelEncoder object for each categorical feature
venue_encoder = LabelEncoder()
batting_team_encoder = LabelEncoder()
bowling_team_encoder = LabelEncoder()
striker_encoder = LabelEncoder()
bowler_encoder = LabelEncoder()
 
# Fit and transform the categorical features with label encoding
X['venue'] = venue_encoder.fit_transform(X['venue'])
X['bat_team'] = batting_team_encoder.fit_transform(X['bat_team'])
X['bowl_team'] = bowling_team_encoder.fit_transform(X['bowl_team'])
X['batsman'] = striker_encoder.fit_transform(X['batsman'])
X['bowler'] = bowler_encoder.fit_transform(X['bowler'])
Train Test Split

We have split the data into training and testing sets. The training set contains 70 percent of the dataset and rest 30 percent is in test set.
X_train contains the training data for your input features.
X_test contains the testing data for your input features.
y_train contains the training data for your target variable.
y_test contains the testing data for your target variable.

# Train test Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
Feature Scaling

We have performed Min-Max scaling on our input features to ensure all the features are on the same scale
Scaling is performed to ensure consistent scale to improve model performance.
Scaling has transformed both training and testing data using the scaling parameters.

from sklearn.preprocessing import MinMaxScaler
 
scaler = MinMaxScaler()
 
# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
Step 4: Define the Neural Network

We have defined a neural network using TensorFlow and Keras for regression.
After defining the model, we have compiled the model using the Huber Loss.

# Define the neural network model
model = keras.Sequential([
    keras.layers.Input( shape=(X_train_scaled.shape[1],)),  # Input layer
    keras.layers.Dense(512, activation='relu'),  # Hidden layer with 512 units and ReLU activation
    keras.layers.Dense(216, activation='relu'),  # Hidden layer with 216 units and ReLU activation
    keras.layers.Dense(1, activation='linear')  # Output layer with linear activation for regression
])
 
# Compile the model with Huber loss
huber_loss = tf.keras.losses.Huber(delta=1.0)  # You can adjust the 'delta' parameter as needed
model.compile(optimizer='adam', loss=huber_loss)  # Use Huber loss for regression
Step 5: Model Training
We have trained the neural network model using the scaled training data.

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test))
Output:

Epoch 1/50
832/832 [==============================] - 4s 4ms/step - loss: 32.9487 - val_loss: 22.0690
Epoch 2/50
832/832 [==============================] - 3s 3ms/step - loss: 22.3249 - val_loss: 22.5012
Epoch 3/50
832/832 [==============================] - 3s 4ms/step - loss: 22.2967 - val_loss: 22.0187
Epoch 4/50
832/832 [==============================] - 3s 4ms/step - loss: 22.2845 - val_loss: 21.9685
Epoch 5/50
832/832 [==============================] - 3s 3ms/step - loss: 22.2155 - val_loss: 21.9134
After the training, we have stored the training and validation loss values to our neural network during the training process.

model_losses = pd.DataFrame(model.history.history)
model_losses.plot()
Output:
![image](https://github.com/surajmhulke/IPL-Score-Prediction-using-Deep-Learning/assets/136318267/a35b01a2-ebe9-4ac6-9a23-c8f911360b8e)

Epoch vs Loss & Validation Loss-Geeksforgeeks
Epoch vs Loss & Validation Loss

Step 6: Model Evaluation

We have predicted using the trained neural network on the testing data.
The variable predictions contains the predicted total run scores for the test set based on the model’s learned patterns.

# Make predictions
predictions = model.predict(X_test_scaled)
 
from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,predictions)
Output:

9.62950576317203
Step 7: Let’s create an Interactive Widget
We have created an interactive widget using ipywidgets to predict the score based on user input for venue, batting team, bowling team, striker, and bowler.
We have created dropdown widgets to select values for venue, batting team, bowling team, striker, and bowler.
Then, we have added a “Predicted Score” button widget. Whenever, the button will be clicked, the predict_score function will be called and then perform the following steps:
Decodes the user-selected values to their original categorical values.
Encodes and scales these values to match the format used in model training.
Uses the trained model to make a prediction based on the user’s input.
Displays the predicted score.

import ipywidgets as widgets
from IPython.display import display, clear_output
 
import warnings
warnings.filterwarnings("ignore")
 
venue = widgets.Dropdown(options=df['venue'].unique().tolist(),description='Select Venue:')
batting_team = widgets.Dropdown(options =df['bat_team'].unique().tolist(),  description='Select Batting Team:')
bowling_team = widgets.Dropdown(options=df['bowl_team'].unique().tolist(),  description='Select Batting Team:')
striker = widgets.Dropdown(options=df['batsman'].unique().tolist(), description='Select Striker:')
bowler = widgets.Dropdown(options=df['bowler'].unique().tolist(), description='Select Bowler:')
 
predict_button = widgets.Button(description="Predict Score")
 
def predict_score(b):
    with output:
        clear_output()  # Clear the previous output
         
 
        # Decode the encoded values back to their original values
        decoded_venue = venue_encoder.transform([venue.value])
        decoded_batting_team = batting_team_encoder.transform([batting_team.value])
        decoded_bowling_team = bowling_team_encoder.transform([bowling_team.value])
        decoded_striker = striker_encoder.transform([striker.value])
        decoded_bowler = bowler_encoder.transform([bowler.value])
 
 
        input = np.array([decoded_venue,  decoded_batting_team, decoded_bowling_team,decoded_striker, decoded_bowler])
        input = input.reshape(1,5)
        input = scaler.transform(input)
        #print(input)
        predicted_score = model.predict(input)
        predicted_score = int(predicted_score[0,0])
 
        print(predicted_score)
The widget-based interface allows you to interactively predict the score for specific match scenarios. Now, we have set up the button to trigger the predict_score function when clicked and display the widgets for venue, batting team , bowling team, striker and bowler.


predict_button.on_click(predict_score)
output = widgets.Output()
display(venue, batting_team, bowling_team, striker, bowler, predict_button, output)
Output:

![image](https://github.com/surajmhulke/IPL-Score-Prediction-using-Deep-Learning/assets/136318267/7e28925f-b709-467d-a728-3b14875fa7a7)
