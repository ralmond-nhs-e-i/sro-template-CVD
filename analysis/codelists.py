from cohortextractor import codelist_from_csv
from config import codelist_path, codelist_system, codelist_code_column

# Change the path of the codelist to your chosen codelist
codelist = codelist_from_csv(codelist_path,
                              system=codelist_system,
                              column=codelist_code_column,)


ethnicity_codes = codelist_from_csv(
        "codelists/opensafely-ethnicity.csv",
        system="ctv3",
        column="Code",
        category_column="Grouping_6",
    )

ld_codes = codelist_from_csv(
    "codelists/opensafely-learning-disabilities.csv",
    system="ctv3",
    column="CTV3Code",
)