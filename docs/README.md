# Design Diagrams

## Already generated (via `docs/generate_diagrams.py`)

- `architecture_diagram.png` — System Architecture Diagram, showing both
  pipelines (recommendation and production/yield) from raw CSV through to
  CLI output.
- `workflow_diagram.png` — Process Flow Diagram for a `predict-yield` call,
  step by step including the validation error branch.

Regenerate either at any time with:
```bash
python docs/generate_diagrams.py
```

## Still to add (UML diagrams)

These need your own module boundaries drawn out — a starting point for
each is given below as Mermaid syntax. Render at https://mermaid.live (or
GitHub's Markdown preview / VS Code's Mermaid extension), export as PNG,
and save into this folder as `use_case_diagram.png`, `class_diagram.png`,
and `sequence_diagram.png`.

### Use Case Diagram

```mermaid
flowchart LR
    Farmer((Farmer /\nAdvisor))
    Analyst((Agri-Planning\nAnalyst))

    UC1([Get crop recommendation\nfor soil & weather conditions])
    UC2([Predict yield & production\nfor a chosen crop])
    UC3([Retrain models on\nupdated data])

    Farmer --> UC1
    Farmer --> UC2
    Analyst --> UC2
    Analyst --> UC3
```

### Class / Component Diagram

```mermaid
classDiagram
    class DataLoader {
        +load_recommendation_data(path) DataFrame
        +load_production_data(path) DataFrame
    }
    class Preprocess {
        +preprocess_recommendation(df) 
        +preprocess_production(df)
        +npk_balance_score(df) Series
        +area_percentile(df) Series
    }
    class ModelTrainer {
        +train_recommend_model(df, config)
        +train_yield_model(df, config)
        +save_model(model, path)
        +load_model(path)
    }
    class Evaluator {
        +evaluate_recommend_model(model, X, y)
        +evaluate_yield_model(model, X, y)
    }
    class Predictor {
        +predict_yield(inputs, config, encoders)
    }
    class Recommender {
        +recommend_crops(inputs, config, encoders)
    }
    class CLI {
        +build_parser()
        +run(argv)
    }

    CLI --> Predictor
    CLI --> Recommender
    Predictor --> ModelTrainer : loads model
    Recommender --> ModelTrainer : loads model
    ModelTrainer --> DataLoader
    ModelTrainer --> Preprocess
    ModelTrainer --> Evaluator
```

### Sequence Diagram — `recommend-crop`

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as cli.py
    participant V as utils.validate_*
    participant R as recommend.py
    participant M as model.py (loaded classifier)

    U->>CLI: python main.py recommend-crop --n 90 --p 42 ...
    CLI->>V: validate N,P,K,temperature,humidity,ph,rainfall
    V-->>CLI: validated values or ValidationError
    CLI->>R: recommend_crops(inputs, config, encoders, top_n)
    R->>R: npk_balance_score() + assemble feature row
    R->>M: model.predict_proba(features)
    M-->>R: class probabilities
    R-->>CLI: top-N {crop, suitability}
    CLI-->>U: JSON result (stdout) + log entry
```
