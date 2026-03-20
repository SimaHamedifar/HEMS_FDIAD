import argparse
import logging
from src.gat_model.train_GAT_LSTM import train
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT-LSTM Forecaster")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--window_size", type=int, default=30, help="Window size (lookback)")
    parser.add_argument("--hidden_dim", type=int, default=16, help="GAT hidden dimension")
    parser.add_argument("--heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--dry_run", action="store_true", help="Run for 1 epoch to test pipeline")
    parser.add_argument("--log_file", type=str, default="training.log", help="Path to log file")
    
    args = parser.parse_args()
    
    try:
        train_loss_list, val_loss_list = train(args)
        
        # Plotting the loss for IEEE Publication
        # Set style to look professional
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 14

        plt.figure(figsize=(6, 4)) # Standard single-column width
        
        # Use distinct styles suitable for b&w printing if needed
        plt.plot(train_loss_list, label='Training Loss', color='b', linestyle='-', linewidth=2)
        plt.plot(val_loss_list, label='Validation Loss', color='r', linestyle='--', linewidth=2)
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Save high quality
        plt.tight_layout()
        plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
        logging.info("Loss plot saved to training_validation_loss.png")
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
