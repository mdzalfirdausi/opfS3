# PGLib-OPF Excel and MAT Conversion Dataset

This repository contains Excel and MAT-converted versions of PGLib-OPF `.m` case files. Each excel file includes additional enhancements, such as an explicit **line ID** field for easier tracking and analysis of transmission lines.

### 📄 Description
- Original data source: [PGLib-OPF](https://github.com/power-grid-lib/pglib-opf)
- Converted from `.m` format to `.xlsx` and `.mat` using a custom parser
- Additional **line ID** column added to the branch data for traceability
- Useful for machine learning, OPF benchmarking, and custom power system analysis in environments outside MATLAB

### 📦 Contents
  - `same_file_name.xlsx`: Excel/MAT file for each PGLib case
  - `bus` sheet
  - `branch` sheet (with line ID)
  - `gen` sheet
  - other standard fields from PGLib-OPF

### 👥 Authors
- Muhammad Dzulqarnain Al Firdausi  
- Mujahid N. Syed

### 📚 Citation
If you use this Excel/MAT dataset in your research, please cite the following:

```bibtex
@misc{firdausi2024pglibexcel,
  author       = {Muhammad Dzulqarnain Al Firdausi and Mujahid N. Syed},
  title        = {PGLib-OPF Excel/MAT Conversion Dataset},
  year         = {2024},
  howpublished = {\url{https://github.com/mdzalfirdausi/opfS3}},
  note         = {Converted from PGLib-OPF case files with additional line ID fields for power system analysis}
}
