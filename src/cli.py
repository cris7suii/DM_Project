"""CLI Inférence Tool — Évaluation en Ligne de Commande d'un Profil Candidat.

Permet d'exécuter des inférences rapides ou des évaluations par lots (batch)
directement depuis le terminal sans nécessiter l'interface graphique Streamlit.

Usage :
    python src/cli.py --techs "Python;Git;TypeScript;Node.js;Bash/Shell" --salary 70000
    python src/cli.py --example
    python src/cli.py --batch data/input.csv --output data/output.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features  # noqa: F401

KEY_TECHS = [
    "Microsoft SQL Server", "npm", "jQuery", "TypeScript", "MongoDB",
    "SQLite", "Bash/Shell", "Node.js", "C#", "Java"
]


def load_artifacts():
    model_p = ROOT / "models" / "best_model.pkl"
    meta_p = ROOT / "models" / "meta.json"
    if not model_p.exists() or not meta_p.exists():
        raise FileNotFoundError("Artefacts introuvables dans models/. Exécutez d'abord les notebooks d'entraînement.")
    model = joblib.load(model_p)
    meta = json.load(open(meta_p, encoding="utf-8"))
    return model, meta


def predict_single(model, profile: dict) -> dict:
    techs_list = [t.strip() for t in profile["HaveWorkedWith"].split(";") if t.strip()]
    key_techs = [t for t in techs_list if t in KEY_TECHS]
    n_keys = len(key_techs)

    row = pd.DataFrame([profile])
    pred = int(model.predict(row)[0])
    dist = float(model.decision_function(row)[0])
    prob = 1.0 / (1.0 + (2.71828 ** (-dist)))

    return {
        "prediction": pred,
        "label": "Employed (1)" if pred == 1 else "Not Employed (0)",
        "hyperplane_distance": round(dist, 4),
        "estimated_probability": round(prob, 4),
        "key_technologies_count": n_keys,
        "key_technologies_detected": key_techs,
        "deterministic_rule_met": (n_keys >= 3),
        "explanation": (
            f"Règle déterministe validée ({n_keys} >= 3 technologies clés)."
            if n_keys >= 3 else
            f"Seuil déterministe non atteint ({n_keys} < 3 technologies clés)."
        )
    }


def main():
    parser = argparse.ArgumentParser(description="EmployML — Outil CLI d'évaluation d'employabilité")
    parser.add_argument("--age", default="<35", choices=["<35", ">35"])
    parser.add_argument("--edlevel", default="Master", choices=["Undergraduate", "Master", "PhD", "NoHigherEd", "Other"])
    parser.add_argument("--mainbranch", default="Dev", choices=["Dev", "NotDev"])
    parser.add_argument("--employment", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gender", default="Man", choices=["Man", "Woman", "NonBinary"])
    parser.add_argument("--country", default="France")
    parser.add_argument("--years-code", type=int, default=8)
    parser.add_argument("--years-pro", type=int, default=5)
    parser.add_argument("--salary", type=float, default=65000.0)
    parser.add_argument("--techs", default="Python;Git;TypeScript;Node.js", help="Technologies séparées par des points-virgules")
    parser.add_argument("--others", type=int, default=2)
    parser.add_argument("--mental", default="No", choices=["No", "Yes"])
    parser.add_argument("--access", default="No", choices=["No", "Yes"])
    parser.add_argument("--batch", type=str, help="Fichier CSV d'entrée pour prédictions par lot")
    parser.add_argument("--output", type=str, help="Fichier CSV de sortie pour les résultats par lot")
    parser.add_argument("--json", action="store_true", help="Format de sortie en JSON")
    parser.add_argument("--example", action="store_true", help="Affiche un exemple d'évaluation complet")

    args = parser.parse_args()
    model, meta = load_artifacts()

    if args.example:
        print("=" * 60)
        print("💼 EmployML — Évaluation d'Exemple Candidat")
        print("=" * 60)
        example_profile = {
            "Age": "<35", "Accessibility": "No", "EdLevel": "Master", "Employment": 1,
            "Gender": "Man", "MentalHealth": "No", "MainBranch": "Dev",
            "YearsCode": 10, "YearsCodePro": 6, "Country": "France", "PreviousSalary": 72000.0,
            "HaveWorkedWith": "Python;Git;TypeScript;Node.js;MongoDB", "ComputerSkills": 7
        }
        res = predict_single(model, example_profile)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if args.batch:
        in_path = Path(args.batch)
        if not in_path.exists():
            print(f"Erreur : fichier {args.batch} introuvable.")
            sys.exit(1)
        df_batch = pd.read_csv(in_path)
        preds = model.predict(df_batch)
        df_batch["Prediction"] = preds
        df_batch["Prediction_Label"] = ["Employed" if p == 1 else "Not Employed" for p in preds]
        out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_predictions.csv")
        df_batch.to_csv(out_path, index=False)
        print(f"Prédictions terminées ({len(df_batch)} lignes). Résultats enregistrés dans : {out_path}")
        return

    techs_clean = [t.strip() for t in args.techs.split(";") if t.strip()]
    profile = {
        "Age": args.age, "Accessibility": args.access, "EdLevel": args.edlevel,
        "Employment": args.employment, "Gender": args.gender, "MentalHealth": args.mental,
        "MainBranch": args.mainbranch, "YearsCode": args.years_code,
        "YearsCodePro": min(args.years_pro, args.years_code),
        "Country": args.country, "PreviousSalary": float(args.salary),
        "HaveWorkedWith": ";".join(techs_clean),
        "ComputerSkills": len(techs_clean) + args.others
    }

    result = predict_single(model, profile)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 50)
        print(f"  RÉSULTAT DE L'INFÉRENCE : {result['label']}")
        print("=" * 50)
        print(f"  - Distance hyperplan SVM : {result['hyperplane_distance']:+.4f}")
        print(f"  - Probabilité estimée    : {result['estimated_probability']:.1%}")
        print(f"  - Technologies clés ({result['key_technologies_count']}/10) : {', '.join(result['key_technologies_detected']) or 'Aucune'}")
        print(f"  - Diagnostic             : {result['explanation']}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
