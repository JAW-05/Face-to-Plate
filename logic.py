import cv2
import numpy as np

def recommend_menu(age, gender, emotion):
    # Swensen's Singapore menu offerings
    menu = {
        "Happy": {
            "Male": [
                "Grilled Fish Sambal", "Battered Fish & Chips", "BBQ Chicken",
                "Ribeye Steak", "Pesto Salmon", "Baked Spaghetti Bolognese"
            ],
            "Female": [
                "Strawberry Shortcake", "Mint Shake Float", "Pesto Salmon",
                "Caesar Salad", "Salmon and Mushroom Baked Rice", "Grilled Chicken"
            ],
            "Unknown": [
                "Mixed Sundae", "Chicken Satay", "Seafood Platter",
                "Fish & Chips", "Pulled Chicken Salad", "Mango Tango Yogurt Smoothie"
            ]
        },
        "Sad": {
            "Male": [
                "Hot Fudge Sundae", "Bangers and Mash", "Chicken Nuggets",
                "Mushroom Soup", "Fish Baked Rice", "Crispy Cornflake Chicken"
            ],
            "Female": [
                "Mango Sorbet", "Choco Lava Cake", "Veggie Wrap",
                "Spicy Chicken Wings", "Curry Chicken Baked Rice", "Pizza"
            ],
            "Unknown": [
                "Fried Rice", "Prawn Noodles", "Clam Chowder",
                "Tom Yum Soup", "Spring Rolls", "Caesar Salad"
            ]
        },
        "Neutral": {
            "Male": [
                "Beef Stroganoff", "Chicken Rice", "Hokkien Mee",
                "Fish & Chips", "Ribeye Steak", "BBQ Chicken"
            ],
            "Female": [
                "Grilled Chicken", "Salmon Salad", "Pasta Carbonara",
                "Oriental Chili Fish Pasta", "Mac N Cheese", "Beef Lasagne"
            ],
            "Unknown": [
                "Yogurt Parfait", "Mocha Madness", "Chocolate Mousse",
                "Seafood Fried Rice", "Creamy Mushroom with Bread Bowl", "Caesar Salad"
            ]
        },
        "Surprise": {
            "Male": [
                "Pork Chop", "Chili Crab", "Baked Ziti",
                "Fish Baked Rice", "Rendang Impossible", "Hawaiian Pizza"
            ],
            "Female": [
                "Beef Tacos", "Sushi Platter", "Stuffed Peppers",
                "Garlic Aioli Impossible", "Thai Curry Pasta", "Pepperoni & Cheese Pizza"
            ],
            "Unknown": [
                "BBQ Ribs", "Pasta Primavera", "Fried Calamari",
                "Fish & Chips", "Clam Chowder", "Creamy Mushroom"
            ]
        },
        "Anger": {
            "Male": [
                "Fried Chicken", "Ramen", "Meatball Sub",
                "Grilled Fish with Meunier", "Spaghetti Carbonara", "Battered Fish & Chips"
            ],
            "Female": [
                "Caesar Wrap", "Grilled Veggie Skewers", "Pasta Pesto",
                "Choco Lava Cake", "Garlic Aioli Impossible", "Crispy Cornflake Chicken"
            ],
            "Unknown": [
                "Sate Platter", "Tandoori Chicken", "Crispy Tofu",
                "Fish Curry", "Seafood Fried Rice", "Pulled Chicken Salad"
            ]
        }
    }

    # Determine the recommendation
    recommendations = menu.get(emotion.capitalize(), {}).get(gender.capitalize(), menu[emotion.capitalize()]["Unknown"])
    return recommendations

def preprocess(image, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image
    