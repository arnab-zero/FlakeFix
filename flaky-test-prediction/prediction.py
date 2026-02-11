"""
Simple Flaky Test Predictor
Quick script to make predictions on Java tests
"""

import torch
from flaky_test_predictor import FlakyTestPredictor, CONFIG

# ==============================================================================
# SETUP - Configure these paths to match your environment
# ==============================================================================

MODEL_CHECKPOINT = r'G:/Flaky Tests/SPL-3/models/model_snapshot_IDoFT.pth'
TRAIN_DATA_FILE = r'G:\Flaky Tests\SPL-3\embeddings\train_embeddings_IDoFT.pt'


# TRAINING_EMBEDDINGS = 'G:/Flaky Tests/SPL-3/embeddings/train_embeddings_IDoFT.pt'
# TRAINING_LABELS = 'G:/Flaky Tests/SPL-3/embeddings/train_labels_IDoFT.pt'

# ==============================================================================
# INITIALIZE ONCE
# ==============================================================================

print("Loading model and embeddings...")
predictor = FlakyTestPredictor(CONFIG, MODEL_CHECKPOINT)

# training_embeddings = torch.load(TRAINING_EMBEDDINGS, map_location=CONFIG['device'])
# training_labels = torch.load(TRAINING_LABELS, map_location=CONFIG['device'])

# Load combined file
train_data = torch.load(TRAIN_DATA_FILE, map_location=CONFIG['device'])

training_embeddings = train_data["embeddings"]
training_labels = train_data["labels"]

if isinstance(training_labels, torch.Tensor):
    training_labels = training_labels.tolist()

predictor.set_class_representatives(training_embeddings, training_labels)
print("✓ Ready to predict!\n")

# ==============================================================================
# FUNCTION TO PREDICT
# ==============================================================================

def predict_test(java_code: str) -> dict:
    """
    Predict if a Java test is flaky.
    
    Args:
        java_code: Java test method as string
        
    Returns:
        {
            'prediction': 'Flaky' or 'notFlaky',
            'confidence': 0.0-1.0,
            'all_confidences': {'notFlaky': X, 'Flaky': Y}
        }
    """
    result = predictor.predict(java_code)
    return result


# ==============================================================================
# DISPLAY RESULT
# ==============================================================================

def display_result(result: dict):
    """Pretty print prediction result."""
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"\nPrediction:  {result['prediction'].upper()}")
    print(f"Confidence:  {result['confidence']:.2%}")
    print(f"\nDetailed Scores:")
    for label, score in result['all_confidences'].items():
        print(f"  • {label:12s}: {score:.4f} ({score*100:6.2f}%)")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    
    # Example test 1
    test1 = """
    @Test
    public void testAsyncOperation() throws InterruptedException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(100);
            return 42;
        });
        
        int result = future.get(5, TimeUnit.SECONDS);
        assertEquals(42, result);
        executor.shutdown();
    }
    """
    
    print("\n[Test 1] Async Operation")
    result1 = predict_test(test1)
    display_result(result1)
    
    # Example test 2
    test2 = """
    @Test
    public void testSimpleAddition() {
        Calculator calc = new Calculator();
        int result = calc.add(5, 3);
        assertEquals(8, result);
    }
    """
    
    print("\n[Test 2] Simple Addition")
    result2 = predict_test(test2)
    display_result(result2)
    
    # You can now use this for your own tests!
    print("\n" + "="*60)
    print("USE THE FUNCTION:")
    print("="*60)
    print("""
    result = predict_test(your_java_test_code)
    print(result['prediction'])      # 'Flaky' or 'notFlaky'
    print(result['confidence'])      # 0.0-1.0
    print(result['all_confidences']) # {'notFlaky': X, 'Flaky': Y}
    """)