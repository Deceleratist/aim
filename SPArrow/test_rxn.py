from run_parallel import run_parallel

#Arguments are r_smiles, p_smiles, job_name, no_duplicates=False, force_depth= False, depth_limit=3, follow_on_enabled=True, aromatic=True, screen_charges_enabled=False, hypervalent_enabled=False

no_duplicates = True
force_depth = True
depth_limit = 5
follow_on_enabled = True
aromatic = True
screen_charges_enabled = False
hypervalent_enabled = True




# --- Initial molecules ---
# sm_smiles = "COC(=O)CC(=O)C.NC(N)=O.c1ccccc1C=O.[OH3+]"
# end_smiles = "COC(=O)C1=C(C)NC(=O)NC1c1ccccc1.O.O.[OH3+]"

# sm_smiles = "O.[CH2+]C=C(C)CCC=C(C)CCC=C(C)C"
# end_smiles = "CC1=CCC23CC1C(C)(C)C2CCC3C.[OH3+]"


# Nenitzescu
# sm_smiles = "[OH3+].CNC=CC(=O)OC.O=C1C=CC(=O)C=C1"
# sm_smiles = "CN[CH+]C(C(=O)OC)C1C=C(O)C=CC1=O.O"
# sm_smiles= "CNC=CC(=O)OC.O.O=C1C=CC(=[OH+])C=C1"
# end_smiles = "O.[OH3+].COC(=O)C1=CN(C)C2=CC=C(O)C=C12"

# gamma-Cadinane
# sm_smiles = "CC(C)=CCCC(C)=CCCC(C)=C[CH2+].O"
# end_smiles = "C=C1CCC(C(C)C)C2=CC(C)CCC12.[OH3+]"

# Biginelli
# sm_smiles = "[OH3+].NC(N)=O.O=CC1=CC=CC=C1.CCOC(=O)CC(C)=O"
# sm_smiles = "CCOC(=O)C=C(C)O.NC(N)=O.O.[OH+]=Cc1ccccc1"
# end_smiles = "CCOC(=O)C1=C(C)NC(=O)NC1c1ccccc1.O.O.[OH3+]"

# sm_smiles ="[OH3+].CC=O.C=CC(=O)C1=CC=CC=C1"
# end_smiles = "O=CCCCC(=O)C1=CC=CC=C1"

# Hantszch
sm_smiles = "COC(=O)C=C(N)C.COC(=O)C(C(=O)C)=Cc1ccccc1.[OH3+]"
end_smiles = "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1.O.[OH3+]"
sm_smiles = "COC(=O)C=C(N)C.COC(=O)C(C(=[OH+])C)=Cc1ccccc1.O"
end_smiles = "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1.[OH3+].O"

# Aldol
# sm_smiles = "CC(=O)CC(=O)OC.O=Cc1ccccc1.[OH3+]"
# end_smiles = "COC(=O)C(=Cc1ccccc1)C(C)=O.O.[OH3+]"

# Aldol
# sm_smiles = "CC(O)=CC(=O)OC.[OH3+].O=Cc1ccccc1"
# end_smiles = "COC(=O)C(=Cc1ccccc1)C(C)=O.O.[OH3+]"


# Heterocycle

# sm_smiles = "c1(C(C)(C)C#N)ccccc1C#N.O.[OH3+].O"
# end_smiles= "CC1(C)C(=[OH+])NC(=O)c2ccccc21.N.O"

# Full DCC

sm_smiles = "C1CCC(N=C=NC2CCCCC2)CC1.C(=O)(CN1C(=O)CCc2ncccc21)O.Nc1scc(Br)c1-c1nccs1"

end_smiles = "C(=O)(CN1C(=O)CCc2ncccc21)Nc1scc(Br)c1-c1nccs1.C1CCC(NC(=O)NC2CCCCC2)CC1"


# Full Mitsunobu

# sm_smiles = "OC1CCN(C(OC(C)(C)C)=O)CC1.Oc1ccc(C#N)cc1.c1ccccc1P(c1ccccc1)c1ccccc1.O=C(N=NC(=O)OCC)OCC"#
# end_smiles = "CC(C)(C)OC(=O)N1CCC(Oc2ccc(C#N)cc2)CC1.CCOC(=O)NNC(=O)OCC.O=P(c1ccccc1)(c1ccccc1)c1ccccc1"

if __name__ == "__main__":
	print("Starting SMILES:", sm_smiles)
	print("Ending SMILES:", end_smiles)
	try:
		run_parallel(sm_smiles, end_smiles, 0, no_duplicates, force_depth, depth_limit, follow_on_enabled, aromatic, screen_charges_enabled, hypervalent_enabled)
	except Exception as e:
		print(e)
		
