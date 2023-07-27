### Usage

1. Set up the configuration parameters in the `config.yaml` file.
2. Execute the main script using the following command:
   ```
   python main.py --config config.yaml
   ```

### Configuration

The configuration parameters for the D-Guard-NLP project are specified in the `config.yaml` file. Modify this file to adjust the settings according to your requirements. The available parameters are as follows:

- `train_csv`: Path to the training dataset in CSV format.
- `test_csv`: Path to the test dataset in CSV format. (Optional)
- `batch_size`: Batch size for training and inference.
- `check_data`: Whether to perform data validation during preprocessing.
- `epochs`: Number of epochs for training.
- `valid_interval`: Interval for validating the model during training.
- `lr`: Learning rate for the optimizer.

Ensure that you have the necessary dependencies installed before running the program.

**Note**: The program supports multi-GPU training using torchrun. If multiple GPUs are available, the program will automatically utilize them for training.

Let's work together to combat fraud with D-Guard-NLP!