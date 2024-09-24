
#** Cat and Dog Classification using Support Vector Machine (SVM)**

This project demonstrates how to implement a Support Vector Machine (SVM) model to classify images of cats and dogs using the Kaggle Cats and Dogs dataset. The goal is to build a robust image classifier that can accurately distinguish between cats and dogs.

## Project Overview

The objective of this project is to apply machine learning techniques to image classification, focusing on the following key steps:

1. **Data Preparation**: Loading, preprocessing, and augmenting the image data.
2. **Feature Extraction**: Converting images into a format suitable for SVM input.
3. **Model Implementation**: Building and training an SVM classifier.
4. **Evaluation**: Testing the model's accuracy and visualizing the results.

## Dataset

The dataset used in this project is the [Kaggle Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data). It consists of 25,000 images of cats and dogs, divided into two classes:

- **Class 1**: Cats
- **Class 2**: Dogs

## Project Structure

The project is organized into the following files and directories:

```
├── data/
│   ├── train/
│   │   ├── cat.0.jpg
│   │   ├── cat.1.jpg
│   │   ├── ...
│   │   ├── dog.0.jpg
│   │   ├── dog.1.jpg
│   │   ├── ...
│   └── test/
│       ├── ...
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_svm_model.ipynb
│   ├── 04_evaluation.ipynb
├── models/
│   ├── svm_model.pkl
├── README.md
└── requirements.txt
```

- **data/**: Contains the training and testing images.
- **notebooks/**: Jupyter notebooks for each step of the project.
- **models/**: Saved SVM model.
- **README.md**: Project documentation.
- **requirements.txt**: Required libraries and dependencies.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/cat-dog-classification-svm.git
   cd cat-dog-classification-svm
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data/` directory.

## Usage

1. **Data Preparation**: Run the `01_data_preparation.ipynb` notebook to load and preprocess the data.

2. **Feature Extraction**: Use the `02_feature_extraction.ipynb` notebook to extract features from the images, such as Histogram of Oriented Gradients (HOG).

3. **Model Training**: Train the SVM model using the `03_svm_model.ipynb` notebook. The model is saved as `svm_model.pkl` in the `models/` directory.

4. **Evaluation**: Evaluate the model's performance with the `04_evaluation.ipynb` notebook, which includes metrics like accuracy, confusion matrix, and ROC curve.

## Results

The trained SVM model achieved an accuracy of **60%** on the test set. The confusion matrix and ROC curve demonstrate the model's ability to classify images of cats and dogs effectively.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Kaggle for providing the [Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
- Scikit-learn for the SVM implementation.

## Contact

If you have any questions, suggestions, or feedback, feel free to reach out:

- **Email**: chibuezechizobafavour@gmail.com




