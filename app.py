import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms
from srgan_model import Generator
import os
import io


# Function to load the pre-trained generator model
def load_generator(model_directory):
    # Automatically search for a model file ending with .pt in the given directory
    model_file = None
    for file in os.listdir(model_directory):
        if file.endswith(".pt"):
            model_file = os.path.join(model_directory, file)
            break

    if not model_file:
        st.error("No model file (.pt) found in the specified directory.")
        st.stop()

    st.write(f"Loading model: `{os.path.basename(model_file)}`")
    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=16)
    generator.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    generator.eval()
    return generator


# Function to preprocess the input image
def preprocess_image(image):
    if image.size[0] > 256 or image.size[1] > 256:
        image = image.resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Function to postprocess the output image
def postprocess_image(output_tensor):
    output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
    output_tensor = torch.clamp(output_tensor, 0, 1)  # Ensure values are in [0, 1] range
    output_image = transforms.ToPILImage()(output_tensor)
    return output_image


# Function to enhance the input image using the generator
def enhance_image(generator, input_image):
    with torch.no_grad():
        enhanced_image, _ = generator(input_image)
    return enhanced_image


# Function to apply strong sharpening to the image
def sharpen_image(image, sharpness_level):
    sharpened_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    enhancer = ImageEnhance.Sharpness(sharpened_image)
    sharpened_image = enhancer.enhance(sharpness_level)
    return sharpened_image


# Function to reduce noise in the image
def reduce_noise(image, noise_reduction_level):
    noise_reduced_image = image.filter(ImageFilter.GaussianBlur(radius=noise_reduction_level))
    return noise_reduced_image


# Streamlit app
def main():
    st.title("Image Enhancement using SRGAN")
    st.write("Upload a low-resolution image to enhance its quality using a pre-trained SRGAN model.")

    # Automatically detect model path
    script_directory = os.path.dirname(__file__)  # Directory of the current script
    model_directory = os.path.join(script_directory, "model")  # Assume model is stored in "model" folder

    # Load generator
    generator = load_generator(model_directory)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert('RGB')

        # Check and resize the image if necessary
        if input_image.size[0] > 256 or input_image.size[1] > 256:
            st.write("Resizing image to 256x256 pixels...")
            resized_input_image = input_image.resize((256, 256))
        else:
            resized_input_image = input_image

        st.write("Enhancing image...")
        input_image_tensor = preprocess_image(resized_input_image)
        enhanced_image_tensor = enhance_image(generator, input_image_tensor)
        enhanced_image = postprocess_image(enhanced_image_tensor)

        st.write("Adjust the sharpness of the enhanced image:")
        sharpness_level = st.slider("Sharpness level", 1.0, 5.0, 2.0, 0.1)

        st.write("Adjust the noise reduction level of the enhanced image:")
        noise_reduction_level = st.slider("Noise reduction level", 0.0, 5.0, 0.0, 0.1)

        st.write("Applying strong sharpening and noise reduction to the enhanced image...")
        sharpened_image = sharpen_image(enhanced_image, sharpness_level)
        noise_reduced_image = reduce_noise(sharpened_image, noise_reduction_level)

        # Convert images to bytes for download
        enhanced_image_bytes = io.BytesIO()
        enhanced_image.save(enhanced_image_bytes, format='PNG')
        enhanced_image_bytes = enhanced_image_bytes.getvalue()

        noise_reduced_image_bytes = io.BytesIO()
        noise_reduced_image.save(noise_reduced_image_bytes, format='PNG')
        noise_reduced_image_bytes = noise_reduced_image_bytes.getvalue()

        # Display images side by side
        col1, col3, col5 = st.columns([1, 1, 1])
        col1.image(resized_input_image, caption='Input Image', use_column_width=False, width=256)
        col3.image(enhanced_image, caption='Enhanced Image', use_column_width=False, width=256)
        col5.image(noise_reduced_image, caption='Final Image', use_column_width=False, width=256)

        # Place download buttons below the images
        st.markdown("---")
        col2, col5 = st.columns([1, 1])
        col2.download_button(
            label="Download Enhanced Image",
            data=enhanced_image_bytes,
            file_name="enhanced_image.png",
            mime="image/png"
        )
        col5.download_button(
            label="Download Final Image",
            data=noise_reduced_image_bytes,
            file_name="final_image.png",
            mime="image/png"
        )


if __name__ == "__main__":
    main()
