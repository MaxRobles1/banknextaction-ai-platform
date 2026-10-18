| Model | Split | Threshold | ROC-AUC | PR-AUC | F1 | Precision | Recall | Accuracy | Floor accuracy | Floor PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| logreg | val | 0.060 | 0.6560 | 0.2332 | 0.2759 | 0.1912 | 0.4955 | 0.7005 | 0.8848 | 0.1152 |
| logreg | test | 0.060 | 0.5813 | 0.4564 | 0.5311 | 0.4376 | 0.6754 | 0.5308 | 0.6066 | 0.3934 |
| mlp (served) | val | 0.555 | 0.6781 | 0.2577 | 0.3115 | 0.2543 | 0.4020 | 0.7953 | 0.8848 | 0.1152 |
| mlp (served) | test | 0.555 | 0.6355 | 0.5094 | 0.5608 | 0.4841 | 0.6664 | 0.5894 | 0.6066 | 0.3934 |

Served model: **mlp** — higher validation F1 (0.3115 vs 0.2759).
