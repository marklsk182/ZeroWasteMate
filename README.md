
#  Project Title: ZeroWasteMate - Freshness Tracker for Eco-Conscious Households

## Introduction 

Among all types of waste in Singapore, food waste is one of the largest waste streams and it is has grown around 20% overthe last 10 years. In 2019, Singapore generated approximately 744 million kilograms of food waste. This significant amount of food waste poses a huge concern, as it necessitates the construction of more disposal facilities, such as incinerators, to reduce this waste to ashes. However, the process of burning food waste requires a substantial amount of heat energy and emits large quantities of carbon dioxide, which are harmful to the environment

Food waste makes up about half of the average waste disposed of each household daily. which more than half of household food waste can be prevented or avoidable, such as expired food, spoil fruits and vegetables and rotten ingredients as well. 

## Background 

Given the high cost of living in Singapore, or due to dietary restrictions, cooking at home becomes a necessity for Singaporeans. With both parents in a household often committed to their work, they might shop for groceries less frequently but in large quantities. This can lead to overbuying, especially of perishable items, which may spoil before they are consumed. Every fresh ingredient has a certain lifespan; purchasing fresh ingredients in large quantities without tracking them can definitely contribute to food wastage, as couples may forget they have these ingredients in their fridge. Moreover, it is a waste of money as well to keep buying and throwing away ingredients when they spoil. 

## Problem statement 

To address the pressing issue of food waste in Singapore, this project proposes the development and implementation of a fresh ingredient recognizer and tracker system. This innovative system aims to reduce avoidable food wastage by leveraging technology to monitor and manage perishable food items. The system will function by:

1. Identifying the freshness level of ingredients using recognition technology.
2. Storing this freshness information in a database.
3. Actively monitoring the shelf life of these ingredients.
4. Sending timely reminders to users about the status of their stored ingredients, including a list of items at risk of spoiling within a certain number of days.

## Objectives  

The goal of this system is to prompt more efficient use of perishable foods, thereby reducing the amount of waste generated due to spoilage. This approach not only seeks to mitigate the environmental impact of food waste but also aims to provide a practical solution for busy households and businesses to manage their food resources more effectively.

## Dataset 

1. Labeled datasets of cabbage with 3 different classes fresh, slightly unfresh and unfresh
2. Labeled datasets of cauliflower with 3 different classes fresh, slightly unfresh and unfresh
3. labeled datasets of red chili with 3 different classes fresh, slightly unfresh and unfresh
4. Labeled datasets of cherry tomatoes with 3 different classes fresh, slightly unfresh and unfresh
5. Labeled datasets of green chili with 2 different classes fresh and unfresh
6. Labeled datasets of tomatoes with 3 different classes fresh, slightly unfresh and unfresh


## Success Metrics 

The primary metric for assessing the success of the ZeroWasteMate is its accuracy in identifying fresh ingredients and determining their freshness levels. Accuracy in this context encompasses two key aspects:

Recognition Accuracy: The system's ability to correctly identify different types of fresh ingredients. This involves distinguishing between various fruits and vegetables

Freshness Level Accuracy: The system's capability to accurately assess the freshness level of each identified ingredient. This requires evaluating the state of the ingredient and estimating how many days remain before it will spoil.

The Second success metrx would be the system's capability to recommend the correct recipes base on the ingredient and it's freshness level  


## Packages and Tools 

1. Tensorflow 2.14.0
2. Tensorflow.keras.preprocessing.image
2. Tensorflow.keras.applications EfficientNetB3/EfficientNetB7/MobileNetV2/InceptionV3
3. Sklearn 1.3.1
4. Sklearn.metrics jaccard_score


---

### Links for background research: 

1. https://www.towardszerowaste.gov.sg/zero-waste-masterplan/chapter3/food/#:~:text=OFF%2DSITE%3A%20TURNING%20FOOD%20WASTE,mixed%20with%20used%20water%20sludge.

2. https://www.sg101.gov.sg/resources/connexionsg/foodwaste/

3. https://www.towardszerowaste.gov.sg/foodwaste/

### Links for dataset research:

1. https://universe.roboflow.com/penulisan-ilmiah-dataset/cabbage-detection

2. https://universe.roboflow.com/leaf-detection-7puag/diseasedleafdetection

3. https://universe.roboflow.com/mseleznova/chilli-pepper

4. https://universe.roboflow.com/maher-9tnii/ripe-tomatoes

5. https://universe.roboflow.com/penelitian-lpgmn/tomato-detection-fresh-or-rotten-using-yolov8

# 1.0 Image Augmentation

## 1.1 Preparation of train dataset 

Due to the limited availability of fresh, slightly unfresh, and unfresh ingredient pictures in my dataset, I aim to ensure that an equal number of pictures is available for each freshness level within every ingredient categories. This balance is crucial to prevent bias during the later stages of model training.

There were six types of ingredients, and each ingredient had three classes of freshness levels, except for Green Chili. Green Chili does not have the 'slightly unfresh' class because green chili naturally progresses to become red chili.

## 1.2 Preparation of test dataset 

For test dataset, aim to have 50 pictures for each freshness level within every ingredient categories.

## 1.3 Preparation of valid dataset 

For validation dataset, aim to have 50 pictures for each freshness level within every ingredient categories.

# 2.0 Exploratory Data Analysis (EDA) 

## 2.1 Identify the number of classes in the dataset

## 2.2 Examine Cabbage dataset 

##### Fresh Cabbage
For fresh cabbage, we typically look for a vibrant and consistent color. Green cabbages, for instance, should appear bright and deep green. The leaves of fresh cabbage should look tight and crisp.

##### Slightly Unfresh Cabbage
Slightly unfresh cabbage may appear duller or have yellowing leaves. The texture of the leaves can appear wilting and loosening. The surface of the cabbage may have some blemishes or spots.

##### Unfresh Cabbage
Unfresh cabbage often shows significant discoloration or has many brown or black spots. The texture of the leaves may look very loose, wilted, or even slimy. The surface may appear rotten or moldy.

## 2.3 Examine Cauliflower dataset 

##### Fresh Cauliflower
Fresh cauliflower is characterized by its firm, tightly packed florets and a uniformly white or creamy color. The florets should be compact and free of any brown or dark spots. Fresh cauliflower also has vibrant green leaves that are crisp and not wilted. The overall appearance should be clean and fresh, with no signs of softness or sogginess.

##### Slightly Unfresh Cauliflower
Slightly unfresh cauliflower the florets might start showing slight discolorations or small brown spots. The cauliflower may have a less vibrant appearance, with the once tightly packed florets beginning to spread or loosen slightly. The leaves may also start to wilt or yellow, and the overall freshness appears diminished.

##### Unfresh Cauliflower
Unfresh cauliflower is easily identifiable by its noticeably discolored, brown, or black spots on the florets. The texture is often soft and soggy, with the florets becoming loose or falling apart.

## 2.4 Examine Cherry Tomatoes dataset 

##### Fresh Cherry Tomatoes
Fresh cherry tomatoes are known for their bright, vibrant color, which can range from a deep red to a rich orange, depending on the variety. They have a smooth and glossy skin. There should be no signs of wrinkling, bruising, or soft spots on the skin.

##### Slightly Unfresh Cherry Tomatoes
Slightly unfresh cherry tomatoes may start to lose their vibrant color, appearing duller. The skin may begin to show slight wrinkling or shriveling, indicating a loss of moisture. They might have minor blemishes or soft spots.

##### Unfresh Cherry Tomatoes
Unfresh cherry tomatoes exhibit significant signs of deterioration. The skin may be deeply wrinkled, discolored, or have multiple soft, bruised spots.

## 2.5 Examine Red Chili dataset 

##### Fresh Red Chili
Fresh red chilies are characterized by their bright, vivid red color. They have a smooth, glossy, and taut skin with a firm texture.The stem of a fresh red chili is typically green and sturdy. There should be no signs of wrinkles, soft spots, or blemishes on the skin.

##### Slightly Unfresh Red Chili
Slightly unfresh red chilies may start to lose their bright color, appearing a bit duller. The skin might begin to show minor wrinkles or slight discoloration. The stem could start to look a bit dry or brown.

##### Unfresh Red Chili
Unfresh red chilies exhibit significant signs of aging or spoilage. The skin may be heavily wrinkled, discolored, or have dark spots. The stem is typically dry, shriveled, or missing.

## 2.6 Examine Tomatoes dataset 

##### Fresh Tomatoes
Fresh tomatoes are characterized by their vibrant, even coloring, which can range from bright red to deep reddish-orange, depending on the variety. They have a firm yet slightly yielding texture when gently squeezed. The skin should be smooth, glossy, and free of blemishes, cracks, or bruises. 

##### Slightly Unfresh Tomatoes
Slightly unfresh tomatoes may begin to lose their bright color, appearing somewhat dull. The skin may start showing signs of wrinkling or slight softening. They might develop small blemishes or soft spots.

##### Unfresh Tomatoes
Unfresh tomatoes show clear signs of overripeness or spoilage. The skin may be deeply wrinkled, discolored, or have multiple soft and bruised spots. These tomatoes are often mushy and yield easily to pressure, indicating a breakdown of the internal structure.

## 2.7 Examine Green Chili dataset 

##### Fresh Green Chili
Fresh green chilies exhibit a vibrant green color, indicative of their freshness and quality. The skin should be smooth, glossy, and taut, without any wrinkles or blemishes. The stem of a fresh green chili is typically bright green and robust. 


##### Unfresh Green Chili
Unfresh green chilies often show significant signs of aging or spoilage. The green color may fade considerably, with pronounced yellowing or brown discoloration. The skin may be heavily wrinkled or have dark spots, and the chili often feels soft or mushy to the touch.

## 2.8 Checking the distribution of the train dataset
From the chart above, it shows that all the six types of ingredients and each classes of freshness levels have the same amount of data.

# 3.0 Data Preprocessing 

##### Training Data Loader (train_generator) 
This loads images from a specified directory, automatically infers labels from the folder structure, resizes images to 224x224 pixels, converts them to categorical format, shuffles the order (with a set seed for reproducibility), and splits the data, reserving 75% for training. It processes the images in batches of 32.

##### Validation Data Loader (val_generator)
This performs the same actions as the training data loader but is specifically for the remaining 25% of the data, designated as the validation set.

These datasets are essential for training and validating a deep learning model, ensuring that it learns from a varied dataset and its performance is evaluated on separate, unseen data.

# 4.0 Building Models 
## 4.1.0 Building model using EfficientNetB3
### 4.1.1 Plotting out the Accuracy and Loss Chart 

## 4.2.0 Building model using EfficientNetB7
### 4.2.1 Plotting out the Accuracy and Loss chart 

## 4.3.0 Building model using MobileNetV2 
### 4.3.1 Plotting out the Accuracy and Loss chart 

## 4.4.0 Building model using InceptionV3
### 4.4.1 Plotting out the Accuracy and Loss chart 

# 5.0 Evaluting model performance
## 5.1 Loading all trained model 
## 5.2 Evaluating EfficientnetB3 Model
## 5.3 Evaluating EfficientnetB7 Model
## 5.4 Evaluating Mobilenetv2 Model
## 5.5 Evaluating InceptionV3 Model
## 5.6 Model performance summary 

|**Model**|**Average val_accuracy**|**Training time**|**Test Accuracy**|**Testing Time**| 
|---|---|---|---|---|
|EfficientNetB3|0.9374|2hr 19min 6s|0.713|52.4s| 
|EfficientNetB7|0.9298|14hr 37min 18s|0.68|2min 51s| 
|MobileNetV2|0.1380|49min 29s|0.109|16.1s| 
|InceptionV3|0.8370|1h 30min 39s|0.651|46.2s|

## 5.7 Confusion Matrix for EfficientnetB3 
From this confusion matrix, we can tell that the model seem to have the tendency to misclassified the ingredients and the freshness level of the ingredients 

# 6.0 Recipe recommender system

After training the model using TensorFlow, the model will output the ingredient name followed by its freshness level. With this information, it can be input into a recommender system, which is then able to generate a few recipes that can be used to cook the ingredient. Now, we will be building several recommender systems and comparing them to determine which is the most trustworthy

## 6.1 Loading of data and perform check on the dataset
## 6.2.0 Recommender system using on Jaccard score
###  6.2.1 create_user_profile function

This function is used to create a user profile based on the liked recipes. In your scenario, instead of using liked recipes, you'll use it to create a profile based on the identified ingredient. For example, if tomatoes_fresh is identified, you'll create a user profile vector where tomatoes_fresh is set to 1 and all other ingredients are set to 0.

###  6.2.2 calculate_similarity function
This function calculates the Jaccard similarity scores between all pairs of recipes in your dataset. It creates a square matrix where the value at row i and column j represents the similarity between recipe i and recipe j based on their ingredients.

###  6.2.3 calculate_similarity function
This function takes a user profile (which could be the vector representing tomatoes_fresh) and the similarity matrix, and it calculates the similarity of the user's preference to each recipe in the matrix. It then sorts these scores and returns the top N recipe IDs as recommendations.

###  6.2.4 Review of recommender system using Jaccard score

When the ingredient 'red chili' and its freshness level of 'slightly unfresh' are detected, the system recommends three recipes. Out of these three, only one recipe, 'Sambal Belacan with Shrimp,' can be used. This is because the rest of the recipes do not contain the ingredient 'red chili' at a 'slightly unfresh' freshness level.

## 6.3.0 Recommender system using cosine similarity 

###  6.3.1 Review of recommender system using cosine similarity

When the ingredient 'red chili' and its freshness level of 'slightly unfresh' are detected, the system recommends three recipes. Out of these three, only one recipe, 'Sambal Belacan with Shrimp,' can be used. This is because the rest of the recipes do not contain the ingredient 'red chili' at a 'slightly unfresh' freshness level.

## 6.4.0  Recommender system using filter-based recommendation

###  6.4.0 Review of recommender system using filter-based recommendation
When the ingredient 'red chili' and its freshness level of 'slightly unfresh' are detected, the system recommends one recipe: 'Sambal Belacan with Shrimp'. This is accurate, as the entire dataset only contains one recipe that can be used based on this ingredient and freshness level.

# 7.0 Key Findings and Limitations

**Key finding for Image Classification Model:**

During the testing phase, I noticed that placing an ingredient on a white background in a well-lit area yields more accurate results compared to positioning it on a non-white background with poor lighting. This may be because ingredients that are not fresh tend to be darker or have a brownish color. Therefore, using a white background with good lighting is sensible for enhancing the accuracy of the results.

**Key finding for Recommendation System:**

During the testing phase, I observed that filter-based recommendations performed better compared to those using Jaccard scores and cosine similarity. With the latter methods, the system often recommended recipes that could not be used because the detected ingredients were not found in the recipe list. However, the filter-based recommendation consistently provided the correct recipes that utilized all of the identified ingredients.

**Limitations of Image Classification Model:**

Given the time constraints of this project, I was only able to source a dataset of fresh ingredients online; there was no time to document the transition of ingredients from fresh to slightly unfresh and then to unfresh. This significant limitation impacted the model's performance, with the best accuracy being only 0.713 (achieved by EfficientNetB3). One key reason for the low model performance might be the lack of a dataset with various freshness levels for each ingredient.

Moreover, most of the ingredient images were taken from an online dataset, which means the size and physical appearance of these ingredients may differ from those we find in local supermarkets. Consequently, the model's performance may decline when tested with local ingredients.


# 7.1 Learning and Future Improvement

**Learning from Image Augmentation and future improvements:**

When augmenting images, try not to augment the same image more than three times for the training dataset, as it may lead to model bias. Always aim to include a greater variety in the images of the ingredients. Ensure a good spread of images with varying lighting conditions; for instance, include pictures of fresh tomatoes taken in well-lit areas as well as in poorly lit areas in the training dataset.

**Learning from Preprocessing Images for modelling and future improvements:**

When using pre-built models from keras.applications, you have the option to use the preprocess_input function from the respective pre-built model that you are utilizing. However, using preprocess_input from the respective pre-built model might cause your model to perform worse during training. Therefore, you can opt to preprocess the data yourself. It is always recommended to first train the model using preprocess_input from the respective pre-built model; if it does not yield good results, then consider preprocessing the data on your own.

**Learning from Building the model and future improvements:**

After training the model, I noticed that using a heavier or larger model does not always yield the best accuracy. Heavier models require significant computing resources for training and also necessitate longer testing times. One area for improvement in modeling would be to select some lightweight and medium-weight models for training as well. 


# 7.2 Conclusion

In conclusion, the ZeroWasteMate project, aimed at reducing food waste in Singapore, yielded valuable insights. Key findings include the superior performance of filter-based recommendations and the importance of image background and lighting in ingredient recognition. The project encountered limitations, notably in dataset diversity and model performance. Future improvements should focus on expanding the dataset and exploring lightweight models. Despite challenges, ZeroWasteMate represents a meaningful step towards environmentally sustainable food management, demonstrating the significant role of technology in ecological conservation.




