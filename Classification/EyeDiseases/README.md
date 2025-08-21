# Eye Disease Classification

This project demonstrates image classification for eye diseases using deep learning. The workflow includes data preparation, model training, evaluation, and saving results.

## Workflow Overview

1. **Data Preparation**

   - Organizes image data into folders by class.
   - Splits data into train, validation, and test sets using stratified sampling.
   - Creates data generators for each set with augmentation for training.

2. **Model Structure**

   - Uses EfficientNetB3 as a pre-trained base model.
   - Adds batch normalization, dense, dropout, and output layers for classification.
   - Compiles with Adamax optimizer and categorical crossentropy loss.

3. **Training**

   - Custom callback manages learning rate, early stopping, and user interaction during training.
   - Plots training history for accuracy and loss.

4. **Evaluation**

   - Evaluates model on train, validation, and test sets.
   - Prints loss and accuracy for each set.

5. **Prediction & Analysis**

   - Predicts classes for test set images.
   - Displays confusion matrix and classification report.

6. **Saving Results**
   - Saves trained model and weights.
   - Generates a CSV file with class indices and image size for reference.

## Key Code Snippet: Save Class Indices & Image Size

```python
class_dict = train_gen.class_indices
img_size = train_gen.image_shape
height = []
width = []
for _ in range(len(class_dict)):
    height.append(img_size[0])
    width.append(img_size[1])

Index_series = pd.Series(list(class_dict.values()), name='class_index')
Class_series = pd.Series(list(class_dict.keys()), name='class')
Height_series = pd.Series(height, name='height')
Width_series = pd.Series(width, name='width')
class_df = pd.concat([Index_series, Class_series, Height_series, Width_series], axis=1)
csv_name = f'{subject}-class_dict.csv'
csv_save_loc = os.path.join(save_path, csv_name)
class_df.to_csv(csv_save_loc, index=False)
print(f'class csv file was saved as {csv_save_loc}')
```

## Requirements

- Python
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- tensorflow (EfficientNetB3)

## Usage

1. Install dependencies:
   ```powershell
   pip install tensorflow pandas numpy scikit-learn seaborn matplotlib
   ```
2. Run the notebook `eye-diseases-classification.ipynb` for step-by-step execution.

## References

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

For details, see the notebook: `eye-diseases-classification.ipynb`.
