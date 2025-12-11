# Tests

This directory is reserved for unit and integration tests.

## Suggested Test Structure

```
tests/
  conftest.py           # Pytest fixtures
  test_data/
    test_loader.py      # Data loading tests
    test_features.py    # Feature engineering tests
    test_normalizer.py  # Normalization tests
  test_models/
    test_analyst.py     # Market Analyst tests
    test_encoders.py    # Encoder tests
    test_fusion.py      # Fusion layer tests
  test_environments/
    test_trading_env.py # Trading environment tests
  test_agents/
    test_sniper_agent.py # Agent tests
  test_integration/
    test_pipeline.py    # Full pipeline tests
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data/test_features.py

# Run with verbose output
pytest tests/ -v
```

## Test Fixtures

Common fixtures should be defined in `conftest.py`:

- `sample_ohlcv`: Small sample OHLCV DataFrame
- `sample_features`: Pre-computed features for testing
- `mock_analyst`: Mock Market Analyst model
- `sample_env`: Pre-configured TradingEnv instance

## Note

This directory is currently empty. Tests should be added incrementally
as the codebase stabilizes. Priority areas for testing:

1. Feature engineering functions (critical for look-ahead bias prevention)
2. Reward calculation in TradingEnv
3. Observation space construction
4. Model inference (context vector generation)
