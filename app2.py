import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd


# Function to load the model
def load_my_model():
    try:
        model = load_model('model_folder/Freshness_classification/efficientnetb3')
        return model
    except Exception as e:
        st.write(f"Error loading the model: {e}")
        return None

# Function to load the data
def load_data():
    ingredient_df = pd.read_excel('recipe_list_2.0.xlsx', sheet_name='ingredients')
    recipes_df = pd.read_excel('recipe_list_2.0.xlsx', sheet_name='recipes')
    return ingredient_df, recipes_df

# Function to create a user profile based on  the input ingredient
def create_user_profile(ingredients, data):
    # Initialize a user profile with zeros
    user_profile = pd.Series(0, index=data.columns)
    # Set the ingredients to 1
    for ingredient in ingredients:
        if ingredient in user_profile.index:
            user_profile[ingredient] = 1
    return user_profile

# Function to recommend recipes by profile
def recommend_recipes_by_profile(user_profile, data):
    # Filter recipes by matching the user profile
    mask = (data * user_profile).sum(axis=1) > 0
    recommended = data[mask]
    return recommended.index

# Class name mapping 
class_names = ['cabbage_fresh', 'cabbage_slightly_unfresh', 'cabbage_unfresh',
               'cauliflower_fresh', 'cauliflower_slightly_unfresh', 'cauliflower_unfresh',
               'cherry_tomatoes_fresh', 'cherry_tomatoes_slightly_unfresh', 'cherry_tomatoes_unfresh',
               'green_chili_fresh', 'green_chili_unfresh', 'red_chili_fresh', 'red_chili_slightly_unfresh',
               'red_chili_unfresh', 'tomatoes_fresh', 'tomatoes_slightly_unfresh', 'tomatoes_unfresh']

# Load the pre-trained model and data
model_efficientnetb3 = load_my_model()
if model_efficientnetb3 is None:
    st.error("Failed to load the model. Please check the model path and try again.")
else:
    ingredient_df, recipes_df = load_data()

    # Streamlit UI
    st.title("Image Classification and Recipe Recommendation")
    # Allow user to choose the number of ingredients
    num_ingredients = st.number_input("Choose the number of ingredients (1-3)", min_value=1, max_value=3, step=1)

    # Initialize list to store the predicted ingredients
    predicted_ingredients = []

    # Function to process and predict each ingredient
    def process_ingredient(upload):
        image = Image.open(upload).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model_efficientnetb3.predict(image_array)
        predicted_class_index = np.argmax(prediction[0])
        return class_names[predicted_class_index], np.max(prediction[0])

    # Loop to handle multiple ingredient uploads
    for i in range(num_ingredients):
        uploaded_file = st.camera_input(f'Take a picture of ingredient {i+1}')
        if uploaded_file:
            predicted_class, confidence = process_ingredient(uploaded_file)
            st.write(f'Predicted Class: {predicted_class}')
            st.write(f'Confidence Score: {confidence:.2f}')

            # Ask the user if the ingredient was wrongly labeled
            wrong_label = st.checkbox("Tick the box if ingredient wrongly labeled.", key=f'wrong_label{i}')

            # If the label is wrong, allow the user to select the correct label
            if wrong_label:
                new_class_index = st.selectbox(
                    "Select the correct ingredient class:",
                    options=range(len(class_names)),
                    format_func=lambda x: class_names[x],
                    key=f'class_select{i}'
                )
                # Replace the current predicted class with the new selection
                predicted_class = class_names[new_class_index]

            # Confirm the ingredient selection
            confirm = st.checkbox("Tick this box to confirm this ingredient", key=f'confirm{i}')
            if confirm:
                # Append the confirmed class
                predicted_ingredients.append(predicted_class)
            else:
                # Handle the case when the user does not confirm the selection
                # Could be a break or continue based on desired flow
                st.warning("Please confirm the ingredient to proceed.")
                continue  # Will move to the next iteration of the loop

    # If ingredients are predicted
    if predicted_ingredients:
        # Create user profile and get recommendations
        user_profile = create_user_profile(predicted_ingredients, ingredient_df)
        recommended_recipe_ids = recommend_recipes_by_profile(user_profile, ingredient_df)
        recommended_recipes = recipes_df.loc[recommended_recipe_ids]

        # Display the recommended recipes in Streamlit
        if not recommended_recipes.empty:
            st.subheader("Recommended Recipes:")
            for _, row in recommended_recipes.iterrows():
                st.markdown(f"<b>{row['recipe_name']}</b>", unsafe_allow_html=True)
                st.write(row['recipe_details'])
        else:
            st.write("No recipes found for the selected ingredients.")

        # Check if any of the predicted classes indicate an inedible (completely unfresh) ingredient
        for predicted_class in predicted_ingredients:
            if '_unfresh' in predicted_class and 'slightly_unfresh' not in predicted_class:
                st.warning(f"⚠️ Be careful with the ingredient {predicted_class.split('_')[0]}. It seems to be unfresh and may not be safe to consume.")