import nbformat as nbf
from config import demographics


nb = nbf.v4.new_notebook()


imports = """\
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from IPython.display import HTML
from IPython.display import Markdown as md
from IPython.core.display import HTML as Center
from utilities import *
from config import marker, start_date, end_date, demographics, codelist_code_column, codelist_term_column, codelist_path


%matplotlib inline


Center('''<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>''')

class Measure:
  def __init__(self, id, numerator, denominator, group_by):
    self.id = id
    self.numerator = numerator
    self.denominator = denominator
    self.group_by = group_by
    
measures = [

    Measure(
        id="total",
        numerator="event",
        denominator="population",
        group_by=["age_band"]
    ),

    Measure(
        id="event_code",
        numerator="event",
        denominator="population",
        group_by=["age_band","event_code"]
    ),

    Measure(
        id="practice",
        numerator="event",
        denominator="population",
        group_by=["age_band","practice"]
    ),


]

for d in demographics:
    if d=='age_band':
        m = Measure(
        id=d,
        numerator="event",
        denominator="population",
        group_by=["age_band"]
        )
        measures.append(m)
    else:
        m = Measure(
            id=d,
            numerator="event",
            denominator="population",
            group_by=["age_band", d]
        )
        measures.append(m)

default_measures = ['total', 'event_code', 'practice']
measures_ids = default_measures+ demographics
measures_dict = {}

for m in measures:
    measures_dict[m.id] = m




"""

header = """\
display(
md("# Service Restoration Observatory"),
md(f"## Changes in {marker} between {start_date} and {end_date}"),
md(f"Below are various time-series graphs showing changes in {marker} code use."),
)
"""

methods = """\
display(
md("### Methods"),
md(f"Using OpenSAFELY-TPP, covering 40% of England's population, we have assessed coding activity related to {marker} between {start_date} and {end_date}. The codelist used can be found here at [OpenSAFELY Codelists](https://codelists.opensafely.org/).  For each month within the study period, we have calculated the rate at which the code was recorded per 1000 registered patients."),
md(f"All analytical code and output is available for inspection at the [OpenSAFELY GitHub repository](https://github.com/opensafely")
)
"""

get_data = """\
default_measures = ['total', 'event_code', 'practice']
measures = default_measures+ demographics

data_dict = {}

for key, value in measures_dict.items():
    
    df = pd.read_csv(f'../output/measure_{value.id}.csv')
    if key == "event_code":
        df.round(16)
    
    to_datetime_sort(df)
    
    if value.id=='age_band':
        data_dict[value.id] = calculate_rate(df, m=value, rate_per=1000, return_age=True)
    elif key == "imd":
       
        df = calculate_rate(df, m=value, rate_per=1000)
       
        df_grouped = calculate_imd_group(df, 'event', 'rate_standardised')
        
        data_dict[value.id] = df_grouped
    
    elif key == "ethnicity":
        df = convert_ethnicity(df)
        data_dict[value.id] = calculate_rate(df, m=value, rate_per=1000)

    else:
        data_dict[value.id] = calculate_rate(df, m=value, rate_per=1000)




codelist = pd.read_csv(f'../{codelist_path}')

"""

output_total_title = """\
display(
md(f"## Total {marker} Number")
)
"""

output_total_plot = """\
plot_measures(data_dict['total'], title=f"Total {marker} across whole population", column_to_plot='rate', category=False, y_label='Rate per 1000')
"""

output_event_codes = """\
display(
md("### Sub totals by sub codes"),
md("Events for the top 5 subcodes across the study period"))
child_table = create_child_table(df=data_dict['event_code'], code_df=codelist, code_column=codelist_code_column, term_column=codelist_term_column)
child_table
    """

output_practice_title = """\
display(
md("## Total Number by GP Practice")
)
"""

output_practice_plot = """\
practice_df = pd.read_csv('../output/input_practice_count.csv')
practices_dict =calculate_statistics_practices(data_dict['practice'], practice_df,end_date)
print(f'Practices included entire period: {practices_dict["total"]["number"]} ({practices_dict["total"]["percent"]}%)')
print(f'Practices included within last year: {practices_dict["year"]["number"]} ({practices_dict["year"]["percent"]}%)')
print(f'Practices included within last 3 months: {practices_dict["months_3"]["number"]} ({practices_dict["months_3"]["percent"]}%)')
interactive_deciles_chart(data_dict['practice'], period_column='date', column='event', title='Decile chart',ylabel='rate per 1000')
"""

nb['cells'] = [
    nbf.v4.new_code_cell(imports),
    nbf.v4.new_code_cell(header),
    nbf.v4.new_code_cell(methods),
    nbf.v4.new_code_cell(get_data),
    nbf.v4.new_code_cell(output_total_title),
    nbf.v4.new_code_cell(output_total_plot),
    nbf.v4.new_code_cell(output_event_codes),
    nbf.v4.new_code_cell(output_practice_title),
    nbf.v4.new_code_cell(output_practice_plot),
    ]

counter = """\
i=0
"""

nb['cells'].append(nbf.v4.new_code_cell(counter))

for d in range(len(demographics)):
    cell_counts = """\
    display(
    md(f"## Breakdown by {demographics[i]}")
    )
    counts_df = calculate_statistics_demographics(df=data_dict[demographics[i]], demographic_var=demographics[i], end_date=end_date, event_column='event')
    counts_df
    """
    nb['cells'].append(nbf.v4.new_code_cell(cell_counts))
    
    cell_plot = """\
    plot_measures(data_dict[demographics[i]], title=f'Breakdown by {demographics[i]}', column_to_plot='rate_standardised', category=demographics[i], y_label='Standardised Rate per 1000')
    i+=1
    """
    nb['cells'].append(nbf.v4.new_code_cell(cell_plot))


nbf.write(nb, 'analysis/SRO_Notebook.ipynb')