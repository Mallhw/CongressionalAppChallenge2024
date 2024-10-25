import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import EnhancedCNN  # Ensure to import your model correctly
from home_screen import show_home  # Import the home screen from the separate file

# Initialize the model
num_classes = 5  # Update based on your severity classes
model = EnhancedCNN()  # Make sure to define your model class
model = torch.load('Model_CNN_FocalLoss20EPOCHS.pt')  # Load model weights
model.eval()  # Set the model to evaluation mode

# Define severity labels
severity_labels = ['None', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# Define the image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert('RGB')  # Open and convert the image to RGB
        image = transform(image)  # Apply the transformation defined above
        image = image.unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def predict_severity(image):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        probabilities = model(image)  # Model should return probabilities

        # Get the predicted class and its probability
        predicted_class = torch.argmax(probabilities, dim=1)
        max_prob = torch.max(probabilities, dim=1).values

    return severity_labels[predicted_class], max_prob.item(), probabilities[
        0].tolist()  # Convert to list for easier usage


def plot_probabilities(probabilities):
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']  # Replace with your actual class names
    plt.figure(figsize=(10, 5))
    plt.bar(classes, probabilities, color=['green', 'yellow', 'orange', 'red', 'purple'])
    plt.ylim(0, 1)  # Probability ranges from 0 to 1
    plt.title('Probability Distribution of Severity Types')
    plt.xlabel('Severity Type')
    plt.ylabel('Probability')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Use Streamlit to display the plot
    st.pyplot(plt)


def convert_to_percentages(float_list):
    return [f"{value * 100:.2f}%" for value in float_list]


# Define the image upload and prediction layout
def show_detection():
    st.title("Diabetic Retinopathy Detection")

    uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(uploaded_file)
        predicted_class, max_prob, class_probabilities = predict_severity(processed_image)
        max_prob_percentage = max_prob * 100

        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Determine confidence level
        if max_prob_percentage < 50:
            confidence_level = "Low Confidence"
            confidence_color = "red"
            guidance_message = ("**Low Confidence:** The model's prediction should be interpreted with caution. "
                                "It is recommended to consult a healthcare professional for further evaluation.")
        elif 50 <= max_prob_percentage < 75:
            confidence_level = "Moderate Confidence"
            confidence_color = "yellow"
            guidance_message = ("**Moderate Confidence:** While the model provides a reasonable prediction, "
                                "it is advisable to monitor your symptoms and consider a consultation with a healthcare provider.")
        else:
            confidence_level = "High Confidence"
            confidence_color = "green"
            guidance_message = (
                "**High Confidence:** The model indicates a strong likelihood regarding your condition. "
                "However, it is still important to discuss your results with a healthcare professional.")

        # Display confidence level with colored text
        st.markdown(f"<h3 style='color: {confidence_color};'>Confidence Level: {confidence_level}</h3>",
                    unsafe_allow_html=True)
        st.markdown(guidance_message)

        st.write(f"Severity Prediction: {predicted_class}")
        st.write(f"Maximum Probability: {max_prob_percentage:.2f}%")
        st.write("Class Probabilities: ", convert_to_percentages(class_probabilities))

        # Plot the probability distribution
        plot_probabilities(class_probabilities)

def show_about():
    st.title("About RetinaGuard")
    st.markdown("""
        **RetinaGuard** was developed to provide accessible diabetic retinopathy detection using advanced convolutional neural networks. 
        The model was trained on thousands of retinal images and optimized for high accuracy.

        ### Key Features:
        - Fast and efficient image processing.
        - Confidence-based predictions with clear severity categories.
        - Easy-to-use interface for quick analysis.

        **Contact Us**: If you have any questions or feedback, feel free to reach out at matthewli2026@gmail.com.
    """)


def show_description():
    try:
        st.title("Description of Diabetic Retinopathy")

        # Add an image at the top
        st.image("dr_diagram.jpg")

        # Enhanced description content
        st.markdown("""
        ### **Diabetic Retinopathy: Causes, Symptoms, Harms, and Risks**

        **Diabetic Retinopathy** is a serious eye condition caused by damage to the blood vessels in the retina, which is triggered by prolonged high blood sugar levels. As a common complication of diabetes, it can progressively impair vision and even lead to blindness if left untreated. The retina, located at the back of the eye, is critical for converting light into visual signals sent to the brain, and damage to this sensitive tissue can result in permanent vision loss.

        ### **Causes**
        The primary cause of diabetic retinopathy is the impact of high blood sugar on the small blood vessels in the retina. Over time, elevated blood sugar levels can lead to the blockage or swelling of these vessels, causing them to leak or form new abnormal vessels. This results in impaired blood flow, retinal damage, and eventually, vision loss.

        ### **Symptoms**
        Early stages of diabetic retinopathy may not present noticeable symptoms, but as the condition progresses, individuals may experience:
        - **Blurred vision**
        - **Dark spots or floaters** that drift in the field of vision
        - **Difficulty seeing colors clearly**
        - **Sudden or gradual loss of vision**

        It is crucial to undergo regular eye exams, especially for individuals with diabetes, as early detection and treatment can significantly reduce the risk of severe outcomes.

        ### **Harms and Risks**
        Without timely intervention, diabetic retinopathy can cause serious complications. These include:
        - **Retinal detachment**, where the retina pulls away from the supportive tissue, leading to permanent vision loss.
        - **Glaucoma**, a condition that increases pressure within the eye, damaging the optic nerve.
        - **Macular edema**, swelling in the central part of the retina that causes distortion and loss of central vision.

        People with poorly controlled diabetes, high blood pressure, or high cholesterol are at greater risk of developing this condition. Early detection through regular eye check-ups, maintaining optimal blood sugar levels, and leading a healthy lifestyle are critical in managing and preventing the progression of diabetic retinopathy.

        For more detailed information, visit the [National Eye Institute](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy#:~:text=Diabetic%20retinopathy%20is%20caused%20by,vessels%20all%20over%20the%20body.).
        """)

    except Exception as e:
        st.error(f"An error occurred while displaying the description: {e}")


def show_challenge():
    st.title("The Growing Challenge of Diabetic Retinopathy")

    st.markdown("""
    ### **Diabetic Retinopathy: A Silent Epidemic**

    Diabetic Retinopathy (DR) has increasingly become a major health concern, affecting millions across the globe. In 2021, **approximately 9.60 million individuals in the United States** were diagnosed with DR, and **1.84 million of these cases** were classified as vision-threatening. According to **Elizabeth A. Lundeen, PhD**, from the **US Centers for Disease Control and Prevention**, this staggering statistic reflects the growing burden of DR within the population, with far-reaching implications for healthcare resources and patient quality of life.

    """)
    st.image("dr_map.jpg", caption="Diabetic Retinopathy Prevalence Map")
    st.image("eye_doctor_map.png", caption="Global Access to Eye Care Professionals")

    st.markdown("""
    ### **Understanding the Root Causes of Diabetic Retinopathy**

    Diabetic Retinopathy arises primarily due to **elevated blood sugar levels**, which damage the intricate blood vessels in the retina, leading to a series of harmful changes, including:

    - **Blood Vessel Blockage**: The initial stages involve the blockage of small blood vessels that supply oxygen to the retina, restricting essential blood flow.
    - **Formation of Abnormal New Vessels**: As a compensatory response, the body attempts to grow new blood vessels. However, these vessels are typically weak and improperly formed.
    - **Leaky Blood Vessels**: These fragile new vessels are prone to leaking, leading to fluid accumulation in the retina, ultimately impairing vision and causing potential blindness.

    ### **Global Impact and Future Projections**

    The worldwide prevalence of diabetes continues to rise sharply, with the number of adults aged 20-79 affected by diabetes projected to increase from **536 million in 2021** to **782 million by 2045**. Among these individuals:

    - **Nearly 60% of those with Type 2 diabetes (T2)** are at risk of developing DR over their lifetime.
    - **Almost all patients with Type 1 diabetes (T1)** will experience some form of DR within **20 years of diagnosis**.

    These projections underscore an urgent public health challenge, as millions of individuals are at risk of severe vision loss and blindness due to DR.

    ### **Challenges in Diagnosis and Treatment**

    Despite the high prevalence of DR, effective diagnosis and treatment remain challenging due to several key factors:

    - **Time-Intensive Diagnostics**: Diagnosing DR accurately requires extensive time and specialized retinal imaging equipment.
    - **Shortage of Trained Ophthalmologists**: Only certified ophthalmologists are capable of providing an official DR diagnosis, and there is a **global shortage** of these professionals, especially in low-resource settings.
    - **Human Error in Diagnosis**: Even among trained specialists, there is a risk of diagnostic error, which can delay treatment and worsen outcomes.

    This combination of **escalating diabetic populations**, **scarcity of healthcare professionals**, and the **critical need for early, precise diagnosis** emphasizes the urgent demand for innovative, accessible solutions to combat the spread of Diabetic Retinopathy globally.
    """)

    st.markdown("""
        ### **Addressing Diabetic Retinopathy in China**

        As a country with a rapidly increasing diabetes prevalence, China faces significant challenges in managing diabetic retinopathy (DR). The following points outline the key issues that need urgent attention:

        - **Rising Diabetes Rates**: China has witnessed a staggering rise in diabetes cases, with millions of individuals currently living with the condition. This rise directly correlates with an increased incidence of DR.

        - **Limited Screening Access**: Despite the growing need, access to screening and diagnostic services for DR remains limited, particularly in rural and underserved areas. This results in many cases going undetected until significant damage has occurred.

        - **Insufficient Medical Resources**: The healthcare system in China struggles with a shortage of trained ophthalmologists capable of diagnosing and treating DR. This gap in expertise leads to delays in patient care and increased risk of vision loss.

        - **Awareness and Education**: There is a pressing need for greater public awareness regarding the importance of regular eye exams and the early signs of DR. Many patients are unaware of the potential consequences of uncontrolled diabetes on their vision.

        - **Cultural Barriers**: Cultural attitudes towards health and healthcare practices can hinder individuals from seeking timely treatment. This can lead to a lack of engagement with preventive measures.

        - **Healthcare Disparities**: Disparities in healthcare access across urban and rural regions exacerbate the challenges faced in managing DR, with urban populations often receiving better care than their rural counterparts.

        ### **Call to Action**

        Addressing these challenges requires a multifaceted approach that includes:

        - Improving access to screening services.
        - Enhancing education and awareness programs about diabetes and DR.
        - Training more healthcare professionals in the detection and treatment of DR.
        - Promoting research and development of innovative solutions to combat DR effectively.

        By focusing on these areas, we can make significant strides in reducing the burden of diabetic retinopathy in China and improve the quality of life for millions.
        """)


def show_cnn_description():
    st.title("Understanding the CNN Model for Diabetic Retinopathy Detection")

    st.markdown("""
    ### **What is a Convolutional Neural Network (CNN)?**

    A **Convolutional Neural Network (CNN)** is a specialized type of deep learning model designed to process data with a grid-like topology, such as images. CNNs are highly effective in tasks like image recognition, classification, and detection due to their ability to detect and learn important visual features directly from images, such as edges, textures, shapes, and other complex patterns. CNNs are composed of several interconnected layers:

    - **Convolutional Layers**: These layers apply filters to the input image to identify essential features (like edges, textures, etc.). Each filter generates a feature map that highlights specific aspects of the image.
    - **Pooling Layers**: Pooling layers reduce the spatial size of the feature maps, maintaining essential information while minimizing computational cost.
    - **Fully Connected Layers**: At the end of the network, fully connected layers interpret the features learned by convolutional layers to classify the image into specific categories, such as 'No DR', 'Mild DR', etc.

    CNNs are structured to learn progressively complex features as the data moves deeper into the network. This makes them particularly suited for identifying subtle details in medical imaging, like signs of diabetic retinopathy in retina images.
    """)
    st.image("cnn_model_example.jpg", caption="An illustration of CNN layers detecting features at different depths.")

    st.markdown("""
    ### **Model Architecture and Functionality**

    In this project, our CNN model consists of four convolutional layers, with each layer progressively extracting more complex features from the retina images. The model architecture includes:

    - **Input Layer**: Images are resized to **256x256 pixels**, allowing the model to process uniform-sized inputs.
    - **Convolutional Layers**: Four convolutional layers with activation functions, each layer extracting increasingly complex visual patterns.
    - **Dropout Layer**: A dropout rate of **0.1** is applied to reduce overfitting, helping the model generalize better to new data.
    - **Output Layer**: The model outputs probabilities for each class, which are then mapped to specific diabetic retinopathy severity levels like 'No DR', 'Mild', 'Moderate', etc.

    This architecture enables the model to perform highly accurate classification based on retinal features, improving diagnostic potential for diabetic retinopathy.

    ### **Dataset Used for Training**

    To train this model, we used a dataset composed of thousands of retina images labeled by experts according to the presence and severity of diabetic retinopathy. Key aspects of the dataset:

    - **Image Labels**: Each image is labeled as one of several categories that represent the severity of DR, such as 'No DR', 'Mild DR', 'Moderate DR', and so on.
    - **Data Augmentation**: Data augmentation techniques like rotation, flipping, and zooming were applied to expand the training data, helping the model become robust against variations in image orientation and lighting.
    - **Balanced Dataset**: The dataset was balanced across classes to ensure that the model learned equally from each DR severity level, reducing any potential bias.

    By training on this diverse and extensive dataset, the CNN model has become adept at identifying features characteristic of diabetic retinopathy, offering a reliable tool for automated diagnosis.

    ### **Performance and Impact**

    The CNN model achieved an accuracy of **86%**, a significant improvement over initial benchmarks. This high accuracy demonstrates the potential of AI-driven tools in assisting ophthalmologists with DR diagnosis, potentially offering more accessible and timely screening for at-risk populations.
    """)


def show_research():
    st.title("Research on Diabetic Retinopathy Detection")

    st.markdown("""
    ### **Dataset Challenges: Imbalance in Severity Classes**

    During the early stages of building a diabetic retinopathy detection model, I identified a significant issue: **class imbalance** in the dataset. The dataset, sourced from **Kaggle**, contained far more examples of some severity levels than others. This imbalance posed a risk for the model, as it could become biased towards predicting the more frequent classes, leading to poor accuracy on underrepresented severity levels.

    ### **Enhancing Image Quality with CLAHE**

    To improve the quality of the images before feeding them into the model, I applied a filter known as **CLAHE (Contrast Limited Adaptive Histogram Equalization)**. This technique helps enhance the contrast of the images, making features more distinct. By applying CLAHE, I ensured that the model could better differentiate between various retinal characteristics associated with different severity levels of diabetic retinopathy.""")
    st.image("Images9_PreCLAHE.png", caption="Examples of Data Augmentation Techniques Applied", use_column_width=True)
    st.image("Images9_PostCLAHE.png", caption="Examples of Data Augmentation Techniques Applied", use_column_width=True)

    st.markdown("""
    ### **Addressing Imbalance with Data Augmentation**

    To tackle the challenge of class imbalance, I implemented **data augmentation** techniques. Data augmentation artificially increases the diversity of the dataset by generating variations of existing images, allowing the model to learn from a more balanced representation of each severity type. Here are the specific techniques I applied:

    - **Rotation**: Slight rotations helped the model become robust to changes in orientation.
    - **Flipping**: Horizontal and vertical flips introduced mirrored perspectives.
    - **Zooming**: Random zoom levels provided different scales of focus on retinal features.
    - **Brightness Adjustment**: Varying brightness allowed the model to learn from images with different lighting conditions.

    By generating new versions of images in the underrepresented classes, I was able to **rebalance the dataset**. This enhancement provided the model with enough examples from each severity level to significantly improve prediction accuracy.
    """)
    st.image("chart.png", caption="Examples of Data Augmentation Techniques Applied", use_column_width=True)
    st.image("Histogram Pre-AUG.png", caption="Examples of Data Augmentation Techniques Applied", use_column_width=True)
    st.image("Histogram POST-AUG.png", caption="Examples of Data Augmentation Techniques Applied", use_column_width=True)
    st.markdown("""
    ### **Transition from ResNet to a Custom Model**

    Initially, I experimented with a **ResNet** architecture as a baseline for my model. While it provided a strong starting point, I realized the need for a more tailored approach to optimize accuracy while avoiding overfitting. This led me to design a custom Convolutional Neural Network (CNN) with specific layers and features that suited my dataset's characteristics.

    #### **Custom CNN Architecture**

    Here’s an overview of the architecture I implemented:

    ```python
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.batch_norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.batch_norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.dropout = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))

        x = x.view(-1, 128 * 14 * 14)  # Flatten with the correct size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)

        # Apply softmax to get probabilities
        probabilities = F.softmax(x, dim=1)
    ```

    ### **Optimizing Accuracy and Avoiding Overfitting**

    In designing this architecture, I implemented several strategies to ensure optimal performance:
    - **Batch Normalization**: This helped stabilize and accelerate training by normalizing the output of each layer.
    - **Dropout Layers**: I used dropout to randomly deactivate a fraction of neurons during training, reducing the risk of overfitting.
    - **Max Pooling**: Max pooling layers were included to progressively reduce the spatial dimensions of the input, maintaining important features while minimizing computational load.

    ### **Training Results After 20 Epochs**

    After training the model for **20 epochs**, the final accuracy for each class was as follows:

    | Class      | Accuracy  |
    |------------|-----------|
    | None       | 96.7%     |
    | Mild       | 61.4%     |
    | Moderate   | 49.3%     |
    | Severe     | 40.8%     |
    | Prolific   | 73.8%     |

    Although these results may not seem optimal at first glance, tweaking the model to specifically detect **diabetic retinopathy (DR) versus no DR** significantly improved accuracy. When focusing on this binary classification, the model demonstrated exceptional performance, particularly in distinguishing between the absence and presence of diabetic retinopathy.
    """)
    st.image("loss_graphs.png")
    st.image("Dr_Comparison.png")

    st.markdown("""
    However, when I adjusted the model for a **binary classification** between **diabetic retinopathy (DR) and no DR**, the data transformed significantly. This modification allowed the model to focus more on differentiating between these two states, thus enhancing its accuracy. The accuracy for the binary classification is summarized below:

    | Class      | Accuracy  |
    |------------|-----------|
    | No DR      | 96.7%     |
    | Has DR     | 95.9%     |

    ### **Limitations of the Study**
    
    Despite the promising findings, several limitations impacted the research:
    - **Small Dataset**: The total number of images used in the dataset was less than **4000**, which can limit the model's ability to generalize.
    - **Origin of Data**: All data came from a single hospital, potentially introducing bias and reducing the model's applicability to broader populations.
    - **Imbalanced Dataset**: The dataset remained imbalanced, with relatively few cases of severe diabetic retinopathy and proliferative diabetic retinopathy (PDR).
    - **Time, Memory, and Compute Constraints**: Limited computational resources impacted the depth and complexity of the models I could effectively train.
    - **Inconsistent Models**: Variability in model performance suggests that further optimization and consistency are needed.

    ### **Conclusion and Future Directions**

    Although data augmentation improved the representation of minority classes, the model's overall performance remained suboptimal. The implementation of **Synthetic Minority Over-sampling Technique (SMOTE)** revealed that synthetic images were not as representative of real images, which affected the model's accuracy. Furthermore, while the application of **Focal Loss** did not enhance accuracy for the minority classes, its potential benefit in combination with data augmentation was noted, as seen with **EfficientNetB7**.

    More experimentation is necessary to refine these techniques, as their effectiveness appears to be class-dependent. Moving forward, it's crucial to explore various strategies that could further improve model performance and ensure reliable diabetic retinopathy detection.
    """)


import requests


def find_nearest_ophthalmologist(address):
    api_key = "YOUR_API_KEY"
    endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    # Parameters for the request
    params = {
        "query": f"ophthalmologist in {address}",
        "key": api_key
    }

    # Make a request to the Google Places API
    response = requests.get(endpoint, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            return results
        else:
            return None
    else:
        st.error("Error fetching data from Google Places API.")
        return None


def show_find_ophthalmologist():
    st.title("Find the Nearest Ophthalmologist")

    address = st.text_input("Enter your address:")

    if st.button("Find Ophthalmologist"):
        if address:
            st.write(f"Searching for ophthalmologists near: **{address}**")
            results = find_nearest_ophthalmologist(address)

            if results:
                st.write("### Nearby Ophthalmologists:")
                for place in results:
                    name = place.get('name')
                    location = place.get('formatted_address')
                    st.write(f"- **{name}**: {location}")
            else:
                st.write("No ophthalmologists found near this address.")
        else:
            st.warning("Please enter a valid address.")

# Adding this page to the main navigation
def main():
    st.sidebar.title("Navigation")
    menu = ["Home", "Diabetic Retinopathy", "Detect DR", "Challenge", "Research", "Find Ophthalmologist", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        show_home()
    elif choice == "Diabetic Retinopathy":
        show_description()
    elif choice == "Detect DR":
        show_detection()
    elif choice == "Challenge":
        show_challenge()
    elif choice == "CNN Model":
        show_cnn_description()
    elif choice == "Research":
        show_research()
    elif choice == "Find Ophthalmologist":
        show_nearest_opthalmologist()
    elif choice == "About":
        show_about()

if __name__ == "__main__":
    main()