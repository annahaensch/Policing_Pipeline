# A Data Science Pipeline for Analyzing Policing Data
    
## Background

This repository is contains the computational elements of [_A Multi-Method Data Science Pipeline for
Analyzing Police Service in the Presence of Misconduct_]() by Anna Haensch, Daanika Gordon, Karin Knudson, and Justina Cheng.


## Setting Up your Environment
Before you get started, you'll need to create a new environment using `conda` (in case you need it, [installation guide here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)). If you use `conda` you can 
create a new environment (we'll call it `my_env`) with

```
conda create --name my_env
```

and activate your new environment, with

```
conda activate my_env
```

#### For Mac Users
If you are using a mac and you plan to run the OCR engine, you will also need to make sure that `tesseract` is added to your path my installing it with homebrew.
```
brew install tesseract
```
#### For Everyone

To run the tools in the libarary will need to install the necessary dependencies. First you'll need to conda install 
`pip` and then install the remaining required Python libraries as follows.

```
conda install pip
conda install -c conda-forge poppler
pip install -U -r requirements.txt
```

Now your environment should be set up to run anything in this library. 

## Parsing the pdf Logs
Parsing the log consists of two steps.  First we use the Tessearct Optical Character Recognition (OCR) tool to turn the scanned pdfs into a tabular parquet file. Next we parse the data in the parquet file to get a csv that matches the original call logs.  If you've already done the OCR part you can skip to step 2.

1. Render the primary pdf documents as parquet files via a script run from the `scripts` directory. *warning: this will take several hours.*
```
cd scripts
python pdf_to_parquet.py 2019
python pdf_to_parquet.py 2020
```

2. Parse and process the parquet into a tabular csv format by running the parser, which should only take a minute or two.

```
cd scripts
python parquet_to_csv.py 2019
python parquet_to_csv.py 2020
```

Please be advised that this pipeline can be buggy and needs some attention, so please feel free to either [raise issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) or [open pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) with bug fixes as you see them. 


