import requests
import pandas as pd

def process_row(sus):
    idx, smiles, name, ld50 = sus

    iupac = get_iupac(smiles, name)
    if iupac and not pd.isna(ld50):
        return (idx, iupac, ld50)
    return None

def get_pubchem_url(smiles):
  domain = "compound"
  # used to be namespace = "smiles"
  namespace = "fastidentity/smiles"
  identifiers = smiles

  in_specify = f"{domain}/{namespace}/{identifiers}"
  op_specify = f"property/IUPACName"
  out_specify = f"TXT"

  url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/{in_specify}/{op_specify}/{out_specify}"

  return url

def get_pubchem_url_from_name(name):
  domain = "compound"
  # used to be namespace = "smiles"
  namespace = "name"
  identifiers = name

  in_specify = f"{domain}/{namespace}/{identifiers}"
  op_specify = f"property/IUPACName"
  out_specify = f"TXT"

  url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/{in_specify}/{op_specify}/{out_specify}"

  return url

def get_iupac(smiles, name):
  url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/iupac_name"

  try:
    page = requests.get(url)

    if not page.status_code in [400, 404, 500] and (not "DOCTYPE" in page.text.strip(" \n")):
      # iupac name
      return page.text.strip(" \n")
  except requests.exceptions.ConnectionError:
    pass


  # if cactus doenst work try pub chem
  page = requests.get(get_pubchem_url(smiles))

  # if pub chem works we good
  if not page.status_code in [400, 404, 500]:
    # iupac name
    return page.text.strip(" \n")
  else:
    # if only one name and not nan
    if (not ", " in str(name)) and not pd.isna(name):
      page = requests.get(get_pubchem_url_from_name(name))

      # if only one output given
      if not page.status_code in [400, 404, 500] and len(page.text.split()) == 1:
        # get iupac name
        return page.text.strip(" \n")


  # if nothing works - rip
  return False