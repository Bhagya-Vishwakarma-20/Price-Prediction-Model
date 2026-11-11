{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      pandas numpy requests matplotlib scipy
      statsmodels scikit-learn xgboost joblib
      prophet holidays
      streamlit plotly
    ]))
  ];
}
