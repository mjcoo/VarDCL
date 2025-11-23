import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import get_model
from config import Config
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_recall_curve, f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_test_data():
    """Load test data with the same format as training data"""
    data = {}
    active_features = Config.TEST_FEATURE_CONFIG['active_features']
    feature_info = Config.TEST_FEATURE_CONFIG['feature_info']
    
    for feature_name in active_features:
        if feature_name not in feature_info:
            raise ValueError(f"Feature '{feature_name}' not found in feature_info configuration")
            
        file_path = os.path.join(Config.TEST_DATA_PATH, feature_info[feature_name]['path'])
        feature_data = np.load(file_path)
        
        data[feature_name] = torch.FloatTensor(feature_data)
        logging.info(f"Loaded test feature {feature_name} with shape: {data[feature_name].shape}")
        
    labels = torch.FloatTensor(np.load(f'{Config.TEST_DATA_PATH}/test_label.npy'))
    
    logging.info(f"Loaded test data - Total samples: {len(labels)}")
    logging.info(f"Class distribution - Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)}")
    return data, labels

def create_test_loader(test_data, test_labels):
    """Create test data loader"""
    test_dataset = TensorDataset(*test_data.values(), test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    return test_loader

def calculate_metrics(outputs, labels):
    """Calculate evaluation metrics"""
    predictions = (outputs >= 0.5).float()
    tp = ((predictions == 1) & (labels == 1)).float().sum()
    fp = ((predictions == 1) & (labels == 0)).float().sum()
    fn = ((predictions == 0) & (labels == 1)).float().sum()
    tn = ((predictions == 0) & (labels == 0)).float().sum()
    
    total = len(labels)
    acc = (tp + tn) / total
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    fpr = fp / (tn + fp + 1e-8)
    fdr = fp / (tp + fp + 1e-8)
    
    try:
        auc = roc_auc_score(labels.cpu().numpy(), outputs.detach().cpu().numpy())
    except:
        auc = 0.5
        
    mcc = matthews_corrcoef(labels.cpu().numpy(), predictions.cpu().numpy())
    
    # Calculate AUPR
    precision_vals, recall_vals, _ = precision_recall_curve(labels.cpu().numpy(), outputs.detach().cpu().numpy())
    aupr = np.trapz(recall_vals, precision_vals)
    
    # Calculate F1
    f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())
    
    metrics = {
        'ACC': acc.item(),
        'AUC': auc,
        'AUPR': aupr,
        'F1': f1,
        'Precision': precision.item(),
        'Recall': recall.item(),
        'MCC': mcc,
        'Specificity': specificity.item(),
        'FPR': fpr.item(),
        'FDR': fdr.item(),
        'TN': tn.item(),
        'FP': fp.item(),
        'FN': fn.item(),
        'TP': tp.item(),
        'Total': total
    }
    
    return metrics

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            *inputs, labels = [x.to(device) for x in batch]
            outputs, _, _, _, _= model(inputs)  # Ignore contrastive loss for testing
            all_outputs.extend(outputs.cpu())
            all_labels.extend(labels.cpu())
    
    all_outputs = torch.tensor(all_outputs)
    all_labels = torch.tensor(all_labels)
    metrics = calculate_metrics(all_outputs, all_labels)
    
    return metrics, all_outputs.numpy()

def save_test_results(predictions, labels, fold=None):
    """Save test results to CSV files"""
    if fold is not None:
        # Save predictions for a specific fold
        predictions_path = os.path.join(Config.TEST_RESULT_PATH, f'test_predictions_fold_{fold}.csv')
        metrics_path = os.path.join(Config.TEST_RESULT_PATH, f'test_metrics_fold_{fold}.csv')
    else:
        # Save best predictions
        predictions_path = os.path.join(Config.TEST_RESULT_PATH, 'best_test_predictions.csv')
        metrics_path = os.path.join(Config.TEST_RESULT_PATH, 'best_test_metrics.csv')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': labels.cpu().numpy(),
        'prediction': predictions
    })
    predictions_df.to_csv(predictions_path, index=False)
    logging.info(f"Predictions saved to {predictions_path}")

def test():
    # Setup device
    device = Config.DEVICE
    logging.info(f"Using device: {device}")
    
    # Load test data
    logging.info("Loading test data...")
    test_data, test_labels = load_test_data()
    test_loader = create_test_loader(test_data, test_labels)
    
    # Initialize model
    logging.info("Initializing model...")
    model, _, _ = get_model()
    model = model.to(device)
    
    # Track best model
    best_auc = -1
    best_metrics = None
    best_predictions = None
    best_fold = None
    
    # Evaluate all folds
    for fold in range(1, Config.CV_CONFIG['n_splits'] + 1):
        model_path = f'{Config.MODEL_PATH}/best_model_fold_{fold}.pt'
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found: {model_path}")
            continue
        
        # Load model
        logging.info(f"\nEvaluating fold {fold} model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate
        metrics, predictions = evaluate_model(model, test_loader, device)
        
        # Save fold results
        save_test_results(predictions, test_labels, fold)
        
        # Print metrics
        logging.info(f"\nFold {fold} Test Results:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        # Track best model
        if metrics['AUC'] > best_auc:
            best_auc = metrics['AUC']
            best_metrics = metrics
            best_predictions = predictions
            best_fold = fold
    
    if best_fold is None:
        raise ValueError("No valid models found for evaluation")
    
    # Print best model results
    logging.info(f"\n\nBest model is from fold {best_fold} with test AUC: {best_auc:.4f}")
    logging.info("Best Model Test Results:")
    for metric, value in best_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save best predictions
    save_test_results(best_predictions, test_labels)
    logging.info(f"\nSaved best model (fold {best_fold}) predictions")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(Config.TEST_RESULT_PATH, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    # Run testing
    test()