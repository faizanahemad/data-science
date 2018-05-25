### Data Source Location

#### Old process for : safe-driver, sherbank-housing, titanic

- For Kaggle projects create a sub-dir under kaggle folder called `data`
- Next download the data of the project you are working on 
- Now create a folder inside data with the same name as the project folder name under kaggle directory
- Unzip your data files in the last step's folder
- For example there is a `sherbank-housing` dir under kaggle so all data for this competition should be inside `data/sherbank-housing`
- For results create a sub-dir data/<project-name>/results
- For each result file name the csv as ipynb notebook name + .csv

#### New Process
- Create a subdir `data` under project folder
- For results create a subdir `results` under the project folder

#### Downloading data under new process
- Install kaggle api `pip install kaggle`
- Under the `competition/data` dir run `kaggle competitions download -c <competition-name> -w -q`
- For unzippin .zip files run in `competition/data` dir `unzip \*.zip`

## Project Structure and guidelines
### Structure
- Each kaggle project should have 3 sub-categories of scripts
- 1st script should be EDA
- 2nd Script should be pre-processing, feature creation, here you preprocess data and store in a csv file to use in 3rd script.
- 3rd script should create ML model based on ideas from EDA
- There can multiples of 2nd and 3rd script type.
- Each preprocessing script should take a csv and create a new one, modifying only the feature variables, In this way you can use the same pre-processing script for both Training and Test cases.

## Misc
### Good Python packages
- Tabulate : [SO](http://stackoverflow.com/questions/18528533/pretty-print-pandas-dataframe)
- [MissingNo: Visualize missing data patterns](https://github.com/ResidentMario/missingno)
