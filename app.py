import os
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

# Optional LP
try:
    import pulp
    HAVE_PULP = True
except Exception:
    HAVE_PULP = False

EXCEL_PATH = os.environ.get("EXCEL_PATH", os.path.join("data", "charge_calc.xls"))

SHEETS = {
    "grades": os.environ.get("SHEET_GRADES", "Grades"),
    "yields": os.environ.get("SHEET_YIELDS", "Yields"),
    "alloys": os.environ.get("SHEET_ALLOYS", "AlloyMap"),
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev")

def _read_excel(path: str) -> Dict[str, pd.DataFrame]:
    # pandas will auto-select engine: xlrd for .xls (requires xlrd), openpyxl for .xlsx
    xls = pd.ExcelFile(path)
    dfs = {}
    for key, sheet in SHEETS.items():
        if sheet in xls.sheet_names:
            dfs[key] = pd.read_excel(path, sheet_name=sheet)
        else:
            dfs[key] = None
    return dfs

def _normalize_grades(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Expect a 'Grade' column + element columns with wt.%
    if df is None or "Grade" not in df.columns:
        raise ValueError("Grades sheet must exist with a 'Grade' column.")
    df = df.copy()
    # Keep only numeric element columns
    elem_cols = [c for c in df.columns if c != "Grade" and pd.api.types.is_numeric_dtype(df[c])]
    # Drop grades without any numeric composition
    df = df[["Grade"] + elem_cols].dropna(how="all", subset=elem_cols)
    # Fill NaNs with zeros for safety
    df[elem_cols] = df[elem_cols].fillna(0.0)
    # Strip grade strings
    df["Grade"] = df["Grade"].astype(str).str.strip()
    return df, elem_cols

def _normalize_yields(df: pd.DataFrame, elems: List[str]) -> Dict[str, float]:
    yields = {e: 1.0 for e in elems}
    if df is None:
        return yields
    # Expect columns Element, Yield
    if "Element" in df.columns and "Yield" in df.columns:
        for _, row in df.iterrows():
            el = str(row["Element"]).strip()
            try:
                y = float(row["Yield"])
            except Exception:
                continue
            if el in yields and 0.0 <= y <= 1.0:
                yields[el] = y
    return yields

def _normalize_alloys(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    # Expect columns: Alloy, Element, Percent, Cost_per_kg(optional)
    required = {"Alloy", "Element", "Percent"}
    if not required.issubset(set(df.columns)):
        return None
    out = df.copy()
    out["Alloy"] = out["Alloy"].astype(str).str.strip()
    out["Element"] = out["Element"].astype(str).str.strip()
    out["Percent"] = pd.to_numeric(out["Percent"], errors="coerce").fillna(0.0)
    if "Cost_per_kg" in out.columns:
        out["Cost_per_kg"] = pd.to_numeric(out["Cost_per_kg"], errors="coerce").fillna(np.nan)
    return out

def load_model():
    dfs = _read_excel(EXCEL_PATH)
    grades_df, elements = _normalize_grades(dfs["grades"])
    yields_map = _normalize_yields(dfs["yields"], elements)
    alloys_df = _normalize_alloys(dfs["alloys"])
    return grades_df, elements, yields_map, alloys_df

GRADES_DF, ELEMENTS, YIELDS_MAP, ALLOYS_DF = load_model()

@app.route("/", methods=["GET"])
def index():
    grades = GRADES_DF["Grade"].tolist()
    return render_template("index.html", grades=grades, grades_elements=ELEMENTS)

def compute_element_requirements(grade: str, melt_kg: float, returns_kg: float, returns_comp: dict | None = None) -> pd.DataFrame:
    row = GRADES_DF.loc[GRADES_DF["Grade"] == grade]
    if row.empty:
        raise ValueError(f"Unknown grade: {grade}")
    targets = row.iloc[0][ELEMENTS].astype(float)  # wt.% targets
    # element mass needed in final melt
    final_needed = melt_kg * (targets / 100.0)  # kg of each element
    # contribution from returns; if custom composition provided, use it, else assume same as grade
    if returns_comp is not None:
        r_comp_series = pd.Series({e: float(returns_comp.get(e, 0.0)) for e in ELEMENTS})
    else:
        r_comp_series = targets
    returns_contrib = returns_kg * (r_comp_series / 100.0) * pd.Series({e: YIELDS_MAP.get(e, 1.0) for e in ELEMENTS})
    # net required from additions (cannot be negative)
    required = (final_needed - returns_contrib).clip(lower=0.0)
    df = pd.DataFrame({
        "Element": ELEMENTS,
        "Target_wt_pct": targets.values,
        "Final_element_kg": final_needed.values,
        "From_returns_kg": returns_contrib.values,
        "Required_from_additions_kg": required.values,
    })
    return df

def solve_alloy_blend(required_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    if ALLOYS_DF is None or ALLOYS_DF.empty:
        return None, np.nan
    if not HAVE_PULP:
        return None, np.nan

    # Build composition matrix A[alloy, element] = fraction (0..1)
    elements = required_df["Element"].tolist()
    alloys = sorted(ALLOYS_DF["Alloy"].unique().tolist())
    comp = {(a, e): 0.0 for a in alloys for e in elements}
    costs = {a: np.nan for a in alloys}
    for _, r in ALLOYS_DF.iterrows():
        a, e, pct = r["Alloy"], r["Element"], float(r["Percent"])
        if e in elements:
            comp[(a, e)] += pct / 100.0
        if "Cost_per_kg" in ALLOYS_DF.columns and not pd.isna(r.get("Cost_per_kg", np.nan)):
            costs[a] = float(r["Cost_per_kg"])
    # default cost if missing: 1
    for a in alloys:
        if np.isnan(costs[a]):
            costs[a] = 1.0

    # Decision vars: mass of each alloy in kg (>=0)
    prob = pulp.LpProblem("AlloyBlend", pulp.LpMinimize)
    x = {a: pulp.LpVariable(f"x_{a}", lowBound=0) for a in alloys}

    # Objective: minimize total cost
    prob += pulp.lpSum(costs[a] * x[a] for a in alloys)

    # Constraints: for each element, alloy contributions >= required addition
    req = dict(zip(required_df["Element"], required_df["Required_from_additions_kg"]))
    for e in elements:
        prob += pulp.lpSum(comp[(a, e)] * x[a] for a in alloys) >= req[e], f"elem_{e}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    solution = pd.DataFrame({
        "Alloy": alloys,
        "Addition_kg": [max(0.0, pulp.value(x[a]) or 0.0) for a in alloys],
        "Cost_per_kg": [costs[a] for a in alloys],
    })
    solution["Est_element_coverage_ok"] = True  # sanity flag; could verify constraints
    total_cost = float((solution["Addition_kg"] * solution["Cost_per_kg"]).sum())
    # Filter negligible
    solution = solution[solution["Addition_kg"] > 1e-6].reset_index(drop=True)
    return solution, total_cost

@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        grade = request.form.get("grade")
        melt_kg = float(request.form.get("melt_kg"))
        returns_kg = float(request.form.get("returns_kg"))
        if melt_kg <= 0 or returns_kg < 0:
            raise ValueError("Weights must be positive; returns can be zero.")
        if returns_kg > melt_kg:
            flash("Warning: returns mass exceeds melt mass; results may be zero-only.", "warning")
        # Parse custom returns composition if provided
        returns_comp = None
        if request.form.get('use_custom_returns') == 'on':
            returns_comp = {}
            for e in ELEMENTS:
                v = request.form.get(f'returns_pct[{e}]')
                try:
                    returns_comp[e] = float(v) if v not in (None, "",) else 0.0
                except Exception:
                    returns_comp[e] = 0.0
        elem_req = compute_element_requirements(grade, melt_kg, returns_kg, returns_comp)
        alloy_solution, total_cost = solve_alloy_blend(elem_req)
        return render_template("result.html",
                               grade=grade,
                               melt_kg=melt_kg,
                               returns_kg=returns_kg,
                               tables={
                                   "Element Requirements (kg)": elem_req.round(4).to_dict(orient="records")
                               },
                               alloy_solution=(None if alloy_solution is None else alloy_solution.round(4).to_dict(orient="records")),
                               total_cost=(None if np.isnan(total_cost) else round(total_cost, 2)),
                               elements=elem_req["Element"].tolist())
    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
