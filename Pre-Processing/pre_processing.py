import pandas as pd
import numpy as np

def load_clean_adm(file_name):
    # This function loads and cleans the MIMIC admissions file
    df_adm = pd.read_csv(file_name)
    df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm = add_next_adm(df_adm)
    df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
    return df_adm

def add_next_adm(df_adm):
    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    df_adm = df_adm.reset_index(drop = True)
    df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
    df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)
    rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN 
    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
    return df_adm

def load_clean_notes(file_name):
    df_notes = pd.read_csv(file_name)
    df_notes_dis_sum = df_notes.loc[df_notes.CATEGORY == 'Discharge summary']
    df_notes_dis_sum_last = (df_notes_dis_sum.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
    assert df_notes_dis_sum_last.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'
    return df_notes_dis_sum_last

def load_clean_merge_dataset(adm_file_name, notes_file_name):
    df_adm = load_clean_adm(adm_file_name)
    df_notes = load_clean_notes(notes_file_name)
    df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME']],df_notes[['SUBJECT_ID','HADM_ID','TEXT']], on = ['SUBJECT_ID','HADM_ID'],how = 'left')
    assert len(df_adm) == len(df_adm_notes), 'Number of rows increased'
    df_adm_notes_clean = df_adm_notes.loc[df_adm_notes.ADMISSION_TYPE != 'NEWBORN'].copy()  
    return df_adm_notes_clean