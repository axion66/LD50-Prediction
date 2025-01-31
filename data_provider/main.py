import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import multiprocessing as mp
from goofyah import process_row, get_iupac
import warnings
import sys

def main():
    #warnings.filterwarnings("ignore")

    try:
        df = pd.read_csv("./cleaned_ld50.csv")

        #"""
        iupacs = []
        for idx in tqdm(range(len(df.index))):
            smiles, name, ld50 = df["SMILES"][idx], df["Name"][idx], df["CATMoS_LD50_mgkg"][idx]
            iupac = get_iupac(smiles, name)

            if len(iupac.split()) != 1:
                raise ValueError(f"Wrong iupac output: {iupac}")

            if iupac and not pd.isna(ld50):
                iupacs.append((idx, iupac, ld50))
        #"""

        """
        iupacs = []
        num_workers = 6
        print(num_workers)

        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_row, [(idx, df["SMILES"][idx], df["Name"][idx], df["CATMoS_LD50_mgkg"][idx]) for idx in range(len(df.index))]), total=len(df)))

        iupacs = [r for r in results if r is not None]
        """

        with open("iupac.json", "w") as f:
            json.dump(iupacs, f)
    
    except KeyboardInterrupt:
        with open("iupac.json", "w") as f:
            json.dump(iupacs, f)
        
        sys.exit()

if __name__ == "__main__":
    main()