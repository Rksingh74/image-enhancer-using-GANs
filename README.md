Image Enhancement using SRGAN
Project Overview 📷
This project enhances low-resolution images using a Super-Resolution Generative Adversarial Network (SRGAN) model. The tool applies SRGAN to generate high-quality images, with additional options for sharpness adjustment and noise reduction. The entire application is built with Streamlit for a user-friendly experience.

Features 🚀
Upload low-resolution images (supports .jpg, .jpeg, .png).
Automatically enhances image resolution using a pre-trained SRGAN model.
Sharpness Adjustment: Fine-tune sharpness levels for the enhanced image.
Noise Reduction: Reduce noise using Gaussian blur filters.
View side-by-side comparison of input, enhanced, and final images.
Download the Enhanced Image and Final Image.

**Project Directory**
image-enhancement-srgan/
│
├── model/                           # Folder for pre-trained model (.pt file)
│   └── pre_trained_model_4488.pt    # SRGAN model (add your .pt file here)
│
├── srgan_model.py                   # SRGAN Generator model definition
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation

Run the application:
streamlit run app.py
