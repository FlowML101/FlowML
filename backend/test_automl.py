"""Quick test of Optuna AutoML"""
import sys
sys.path.insert(0, '.')

from services.optuna_automl import optuna_automl

print("Testing Optuna AutoML...")
print("=" * 50)

result = optuna_automl.train(
    dataset_path='uploads/sample_classification.csv',
    target_column='target',
    job_id='test-001',
    time_budget_minutes=1,
    n_trials_per_model=3,
)

print(f"\nBest model: {result.best_model}")
print(f"Models trained: {len(result.models)}")
print(f"Total time: {result.total_time:.1f}s")
print("\nLeaderboard:")
for m in result.models:
    acc = m['metrics'].get('accuracy', 0)
    print(f"  {m['rank']}. {m['algorithm']}: accuracy={acc:.3f}")
