# GA-RAG-Optimizer

Forschungsartefakt zur Forschungsarbeit:
**Evolutionäre Hyperparameter-Optimierung für RAG-Pipelines: Ein Empirischer Vergleich mit klassischen Suchstrategien**

Autor: Marco Karic · Betreuer: Dr.-Ing. Eric MSP Veith · WBH, Studiengang KI und Data Science

---

## Forschungsfrage

> Inwiefern können genetische Algorithmen den gemischten Hyperparameterraum einer RAG-Pipeline effizienter optimieren als Random Search, Grid Search und Default-Konfigurationen, gemessen an referenzbasierten Retrieval-Metriken?

---

## Architektur

```
experiment_config.yaml
        │
        ▼
  experiment_runner/run.py
        │
   ┌────┴────────────────────────────┐
   │                                 │
   ▼                                 ▼
ga_engine/                      baselines/
(DEAP: Turnierselektion,        (Random Search,
 Mixed-Type-Crossover,           Grid Search,
 typspez. Mutation)              Default-Config)
   │                                 │
   └────────────┬────────────────────┘
                ▼
          rag_pipeline/
    (LangChain + ChromaDB,
     Embedding-Modell-Wechsel)
                │
                ▼
          evaluation/
    (Context Recall, Answer-F1,
     Context Precision, Exact Match)
                │
                ▼
           results/
```

---

## Hyperparameterraum

| Hyperparameter      | Typ        | Wertebereich                                          |
|---------------------|------------|-------------------------------------------------------|
| `chunk_size`        | Diskret    | 128, 256, 512, 1024 Zeichen                           |
| `chunk_overlap`     | Diskret    | 0, 32, 64, 128 Zeichen                                |
| `top_k`             | Diskret    | 2, 3, 5, 10                                           |
| `embedding_model`   | Kategorial | all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5 |
| `retrieval_strategy`| Kategorial | Dense, BM25, Hybrid                                   |

GA-Konfiguration: 20 Individuen × 30 Generationen, also höchstens 620 Evaluationen pro Lauf (600 Nachkommen plus die 20 Individuen der Startpopulation). Die tatsächliche Zahl liegt darunter, weil `eaSimple` nur die Nachkommen neu bewertet, die Kreuzung oder Mutation getroffen hat, und steht im Ergebnis als `n_evaluations`. Drei unabhängige Läufe mit festen Seeds.

---

## Setup

```bash
# Virtuelle Umgebung anlegen
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS / Colab

# Abhängigkeiten installieren
pip install -e ".[dev]"

# Jupyter-Kernel für das venv registrieren (einmalig, für Notebooks)
python -m ipykernel install --user --name ga-rag-venv --display-name "Python (GA-RAG venv)"
```

### Evaluations-Datensatz beschaffen (einmalig)

Notebook `notebooks/01_build_eval_dataset.ipynb` in VS Code oder Jupyter öffnen, Kernel auf "Python (GA-RAG venv)" stellen und alle Zellen ausführen.

Erzeugt die Artefakte unter `data/raw/`, `data/processed/`, `data/corpus/` mit festem Seed 42, deterministisch reproduzierbar. Headless lässt sich das Notebook auch via `jupyter nbconvert --to notebook --execute --inplace notebooks/01_build_eval_dataset.ipynb` aufrufen (z. B. in CI oder Colab).

---

## Nutzung

```bash
# Experiment starten
python src/experiment_runner/run.py --config configs/experiment_config.yaml

# Tests
python -m pytest tests/

# Linting & Formatierung
ruff check src/
ruff format src/
```

---

## Läufe starten

Die offiziellen Läufe werden über die Skripte in `scripts/` gestartet, nicht aus einem Notebook. Jeder Aufruf darin startet einen eigenen Python-Prozess. Dadurch teilen sich die Läufe keinen Zustand, also keine geladenen Embedding-Modelle, keine offenen Logging-Handles und keine ChromaDB-Verbindungen. Bricht ein Lauf ab, bleiben die bereits abgeschlossenen unversehrt liegen und das Skript stoppt über den Exit-Code. Die Skripte setzen außerdem `HF_HUB_OFFLINE=1`, bevor Python startet, damit kein Embedding-Modell aus dem Netz nachgeladen wird.

Reihenfolge:

1. `notebooks/04_prebuild_embedding_cache.ipynb` ausführen, damit die ChromaDB-Collections vorliegen.
2. Einen Smoke-Lauf fahren und die Zeit je Evaluation im Log prüfen.
3. Die vier Blöcke starten, jeder einzeln und nacheinander.
4. `notebooks/05_baseline_comparison.ipynb` für Auswertung und Methodenvergleich ausführen.

| Skript | Läufe | Zweck |
|---|---|---|
| `run_smoke.ps1` | 4 | Alle Methoden mit Kleinstwerten, prüft nur die Lauffähigkeit |
| `run_smoke_medium.ps1` | 4 | 20 Fragen und 24 Trials, dient der Laufzeitschätzung |
| `run_block_ga.ps1` | 3 | GA mit den Seeds 42, 43, 44 |
| `run_block_random.ps1` | 3 | Random Search mit den Seeds 42, 43, 44 |
| `run_block_grid.ps1` | 1 | Grid Search, Seed 42, deterministisch |
| `run_block_default.ps1` | 1 | Default-Konfiguration, Seed 42, deterministisch |

```powershell
.\scripts\run_smoke.ps1
```

```powershell
.\scripts\run_block_ga.ps1
```

Zur Laufzeit: Aus den Smoke-Läufen hochgerechnet liegt eine Evaluation über 100 Fragen bei etwa zwölf Sekunden, ein GA-Block mit drei Seeds damit im Bereich von sechs Stunden. Für alle vier Blöcke zusammen ist mit zwölf bis fünfzehn Stunden zu rechnen. Das ist eine Schätzung aus wenigen Fragen und keine Messung, deshalb steht `run_smoke_medium.ps1` in Schritt 2.

---

## Ergebnisse der offiziellen Läufe

Die offiziellen Läufe vom 15.08.2026 liegen in zwei Teilen vor. Die Tabellen unter `results/tables/` und die Abbildungen aus Notebook 05 unter `results/figures/` sind Teil des Repos. Die vollständigen Logs aller Evaluationen (JSONL, entpackt rund 96 MB) hängen als ZIP am [Release v1.0.0](https://github.com/karcomaric/ga-rag-optimizer/releases/tag/v1.0.0).

Um die Auswertung ohne neue Läufe nachzuvollziehen, das ZIP im Wurzelordner des Repos entpacken. Die Logs liegen danach unter `results/logs/ga_rag_optimizer_v1/`, und `notebooks/05_baseline_comparison.ipynb` lässt sich direkt ausführen.

---

## Projektstruktur

```
ga-rag-optimizer/
├── configs/
│   └── experiment_config.yaml   # Suchraum, GA-Parameter, Baselines
├── notebooks/
│   ├── 01_build_eval_dataset.ipynb        # Dataset-Beschaffung (interaktiv, Single Source)
│   ├── 02_evaluation_smoketest.ipynb      # Smoke-Test der Eval-Metriken
│   ├── 03_pipeline_validation.ipynb       # Validierung der LangChain-RAG-Pipeline
│   ├── 04_prebuild_embedding_cache.ipynb  # ChromaDB-Cache vorbauen, läuft vor den vollen GA-/Baseline-Läufen
│   └── 05_baseline_comparison.ipynb       # Auswertung der abgeschlossenen Läufe, Methodenvergleich
├── src/
│   ├── ga_engine/               # DEAP-basierter GA
│   ├── baselines/               # Random Search, Grid Search, Default-Config
│   ├── rag_pipeline/            # LangChain + ChromaDB
│   ├── evaluation/              # Context Recall, Answer-F1
│   ├── experiment_runner/       # Einstiegspunkt run.py
│   ├── schemas/                 # Dataclasses für den Modul-Datenaustausch
│   ├── environment/             # Pfade und Config für Notebooks und Skripte
│   └── visualization/           # Ergebnis-Plots
├── scripts/                     # PowerShell-Runner für Experiment-Blöcke
├── tests/                       # pytest-Tests
├── data/                        # Eval-Dataset (committed), Laufzeit-Caches (gitignored)
│   ├── raw/                     # HotpotQA-Rohstichprobe
│   ├── processed/               # qa_pairs.json, qa_pairs_split.json, corpus_meta.json
│   └── corpus/                  # ~1500 Paragraph-.txt-Dateien
├── results/                     # Tabellen und Abbildungen, Logs als ZIP am Release v1.0.0
└── pyproject.toml               # Abhängigkeiten, ruff, pytest
```

---

## Datensatz

**HotpotQA Distractor** (Validation-Split, 150 Fragen, stratifiziert nach `bridge`/`comparison`-Typ).

- **Train-Pool:** 100 Fragen → GA-Fitness und Baseline-Suche
- **Test-Pool:** 50 Fragen → Nachbewertung der besten Konfiguration je Lauf (Notebook 05, Schritt 9)
- **Korpus:** ~1500 Paragraphen aus den Distractor-Kontexten, eine `.txt`-Datei pro Paragraph (`data/corpus/`)
- **Ground Truth:** `supporting_facts` auf Satz-Ebene, Context-Recall auf Absatzebene ohne LLM

Beschaffung über das Notebook `notebooks/01_build_eval_dataset.ipynb` (einmalig, danach im Repo). Auswahl und Begründung der Datensatzwahl siehe Paper Kap. 2.5.

Die Dateien unter `data/` sind ein Auszug aus HotpotQA (Yang et al., EMNLP 2018, <https://hotpotqa.github.io>) und stehen wie das Original unter der Lizenz [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Die Absätze stammen ursprünglich aus der englischsprachigen Wikipedia.

---

## Infrastruktur

| Umgebung     | Details                                              |
|--------------|------------------------------------------------------|
| Lokal        | VS Code + `.venv`, Python ≥ 3.11                     |
| Checkpoints  | Bewusst nicht implementiert, ein Abbruch startet neu, der Embedding-Cache begrenzt den Schaden |
| Embedding-Cache | Identische Preprocessing-Schritte werden gecacht, vorab gefüllt über `notebooks/04_prebuild_embedding_cache.ipynb` |

---

## Lizenz

Für den Quellcode ist keine Lizenz vergeben. Er ist zur Einsicht im Rahmen der Begutachtung der Forschungsarbeit veröffentlicht, alle Rechte liegen beim Autor. Für die Daten unter `data/` gilt CC BY-SA 4.0, siehe Abschnitt Datensatz.
