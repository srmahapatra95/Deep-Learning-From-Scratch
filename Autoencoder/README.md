# Autoencoder Project

This project implements a Fully Convolutional Autoencoder using PyTorch for image reconstruction tasks. The model is designed to compress input images into a latent representation and then reconstruct them back to the original dimensions.

## Project Structure

*   `AutoEncoders.py`: The main script containing the model architecture (`AutoEncoder`, `Encoder`, `Decoder`), training loop, and evaluation logic.
*   `test.ipynb`: A Jupyter Notebook for testing the trained model, performing inference on new images, and visualizing the results.
*   `ae_checkpoints_fcn/`: Directory where model checkpoints (`ae_best_fcn.pt`) and training reconstruction visualizations are saved.
*   `test_image_data/`: Directory used by the notebook to store downloaded test images.

## Requirements

To run this project, you need the following Python libraries:

*   Python 3.x
*   PyTorch
*   Torchvision
*   Pillow (PIL)
*   python-dotenv
*   Matplotlib
*   Requests

You can install the necessary dependencies using pip:

```bash
pip install torch torchvision pillow python-dotenv matplotlib requests
```

## Setup

1.  **Clone the repository** (if applicable) or navigate to the project directory.
2.  **Environment Variables**: Create a `.env` file in the root directory and specify the path to your training image dataset:
    ```
    IMAGE_FOLDER=/path/to/your/image/dataset
    ```

## Usage

### Training

To train the autoencoder, run the `AutoEncoders.py` script:

```bash
python AutoEncoders.py
```

**Configuration parameters in `AutoEncoders.py`:**
*   `NUM_EPOCHS`: Number of training epochs (default: 30)
*   `LEARNING_RATE`: Learning rate for the Adam optimizer (default: 1e-3)
*   `BATCH_SIZE`: Batch size for data loading (default: 32)
*   `LOG_DIR`: Directory for saving checkpoints and logs (default: `./ae_checkpoints_fcn`)

The script will automatically detect if CUDA or MPS (Apple Silicon) is available and use it for training. It saves the best model as `ae_best_fcn.pt` and reconstruction images every epoch in the `LOG_DIR`.

### Testing & Inference

Open `test.ipynb` in Jupyter Notebook or JupyterLab to interactively test the model. The notebook includes steps to:
1.  Load the trained model architecture and weights.
2.  Download a sample image from the web.
3.  Preprocess the image.
4.  Run the image through the autoencoder.
5.  Visualize the original and reconstructed images.

## Model Architecture

The model consists of two main parts:

*   **Encoder**: A series of Convolutional layers (`Conv2d`) followed by ReLU activations. It progressively reduces the channel dimensions to compress the input.
*   **Decoder**: A series of Transposed Convolutional layers (`ConvTranspose2d`) followed by ReLU activations, mirroring the encoder to upsample the latent representation back to the original image size. The final layer uses a Sigmoid activation to output pixel values between 0 and 1.

Input image size is expected to be resized to **224x224**.
