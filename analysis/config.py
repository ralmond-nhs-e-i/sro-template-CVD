#study start date.  should match date in project.yaml
start_date = "2021-06-01"

#study end date.  should match date in project.yaml
end_date = "2021-12-01"

#demographic variables by which code use is broken down
#select from ["sex", "age_band", "region", "imd", "ethnicity", "learning_disability"]
demographics = ["sex", "age_band", "region", "imd", "ethnicity", "learning_disability"]

#name of measure
marker="ACE Inhibitors"

#codelist path
codelist_path = "codelists/opensafely-atrial-fibrillation-clinical-finding.csv"

#column name referencing code in chosen codelist
codelist_code_column="CTV3Code"

#codelist system for chosen codelist
codelist_system = "ctv3"

#column name referencing code descriptions in chosen codelist
codelist_term_column='Description'