# Smoother

The `Smoother` class provides data smoothing techniques for tabular datasets.

---

## Methods

### `winsorization(df, limits=(0.01, 0.01), columns=None)`

Applies **Winsorization** to numerical columns by capping extreme values instead of removing them.

### Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `df` | `pd.DataFrame` | Required | Input DataFrame. |
| `limits` | `tuple` | `(0.01, 0.01)` | Fraction of observations to cap on the lower and upper ends. |
| `columns` | `list` | `None` | Numerical columns to winsorize. If `None`, all numerical columns are used. |

### Returns

- **Type:** `pd.DataFrame`
- A new DataFrame with Winsorized values.

---

## Example 1 — Winsorize All Numerical Columns

```python
smoother = Smoother()

df_winsor = smoother.winsorization(
    df,
    limits=(0.01, 0.01)   # Cap lowest/highest 1%
)
```

---

## Example 2 — Winsorize Specific Columns

```python
smoother = Smoother()

df_winsor = smoother.winsorization(
    df,
    limits=(0.05, 0.05),
    columns=[
        "age",
        "creatinine_avg",
        "glucose_avg"
    ]
)
```

---

## Notes

- The original DataFrame is **not modified**.
- If `columns=None`, every numerical column is Winsorized.
- Non-numeric columns are left unchanged.
- Winsorization caps extreme values rather than removing rows, preserving the number of observations.